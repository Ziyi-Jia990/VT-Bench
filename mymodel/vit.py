import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import timm
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             mean_squared_error, mean_absolute_error, r2_score)
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import copy

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='min', save_path='checkpoint/best_model.pth'):
        """
        Args:
            patience (int): 在指定次数内指标未改善则停止训练
            min_delta (float): 指标改善的最小阈值
            mode (str): 'min' (如 loss/rmse) 或 'max' (如 accuracy/r2)
            save_path (str): 最佳模型权重的保存路径
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # 确保 checkpoint 目录存在
        checkpoint_dir = os.path.dirname(self.save_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
        else:
            if self.mode == 'min':
                improved = current_score < (self.best_score - self.min_delta)
            else:
                improved = current_score > (self.best_score + self.min_delta)

            if improved:
                self.best_score = current_score
                self.save_checkpoint(model)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, model):
        """将模型权重保存到硬盘"""
        torch.save(model.state_dict(), self.save_path)

# --- 1. 自定义数据集类：处理路径文件 ---
class ImageDataset(Dataset):
    def __init__(self, paths_pt, labels_pt, transform=None):
        self.img_paths = torch.load(paths_pt)
        self.labels = torch.load(labels_pt)
        self.transform = transform
        
        if isinstance(self.labels, torch.Tensor):
            self.labels = self.labels.numpy()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        # --- 针对 .npy 文件的特殊处理 ---
        if img_path.endswith('.npy'):
            # 1. 直接加载 numpy 数组
            img_array = np.load(img_path)
            # 2. 转换为 torch.Tensor
            image = torch.from_numpy(img_array).float()
            
            # 3. 维度检查
            # 如果 .npy 存的是 (H, W, C)，需要转为 ViT 要求的 (C, H, W)
            if image.shape[-1] == 3 and image.ndim == 3:
                image = image.permute(2, 0, 1)
            
            # 注意：因为已经归一化过了，这里不再应用 self.transform
            # 如果你的 .npy 只是缩放了但没转成 C,H,W，上面的 permute 就很重要
        
        # --- 针对普通图像文件的处理 ---
        else:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        
        return image, torch.tensor(label)
    
# --- 2. 训练与验证单轮逻辑 ---
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, task):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if task == 'regression':
            labels = labels.float().view(-1, 1)
            loss = criterion(outputs, labels)
        else:
            loss = criterion(outputs, labels.long())
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    return running_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, task, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        
        if task == 'regression':
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
        else:
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if task == 'classification':
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        if num_classes == 2:
            auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        return {"acc": acc, "auc": auc, "f1": f1}
    else:
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        return {"rmse": rmse, "mae": mae, "r2": r2}

# --- 3. 核心实验运行函数 (超参搜索) ---
def run_vit_experiment(cfg: DictConfig, train_data_info, val_data_info, seed, target):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = cfg.get('task', 'classification')
    num_classes = cfg.get('num_classes', 1 if task == 'regression' else 2)

    # 数据准备 (保持不变)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = ImageDataset(train_data_info['paths'], train_data_info['labels'], transform=transform)
    val_dataset = ImageDataset(val_data_info['paths'], val_data_info['labels'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 搜索范围
    lrs = [5e-4, 1e-3, 3e-3]
    wds = [0.03, 0.1, 0.3]
    epochs_list = [50, 100, 500]

    global_best_val_score = -float('inf') if task == 'classification' else float('inf')
    global_best_results = None
    global_best_params = {}

    for lr in lrs:
        for wd in wds:
            for max_epochs in epochs_list:
                print(f"--- 尝试配置: LR={lr}, WD={wd}, MaxEpochs={max_epochs} ---")
                
                model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes).to(device)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                
                # Warmup 设置
                total_steps = len(train_loader) * max_epochs
                scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=int(0.1 * total_steps))
                criterion = nn.MSELoss() if task == 'regression' else nn.CrossEntropyLoss()

                # 为当前超参组合定义唯一的保存路径
                current_ckpt_path = f"/mnt/hdd/jiazy/checkpoint/{target}/vit_s{seed}_lr{lr}_wd{wd}_ep{max_epochs}.pth"
                
                early_stopper = EarlyStopping(
                    patience=5, 
                    mode='max' if task == 'classification' else 'min',
                    save_path=current_ckpt_path
                )

                for epoch in range(max_epochs):
                    _ = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, task)
                    metrics = evaluate(model, val_loader, device, task, num_classes)
                    
                    current_score = metrics['acc'] if task == 'classification' else metrics['rmse']
                    early_stopper(current_score, model)

                    if early_stopper.early_stop:
                        print(f"      [早停] 触发于 Epoch {epoch}。")
                        break
                
                # --- 核心修改：从硬盘加载该配置下的最佳权重 ---
                if os.path.exists(current_ckpt_path):
                    model.load_state_dict(torch.load(current_ckpt_path))
                
                final_metrics = evaluate(model, val_loader, device, task, num_classes)

                # 更新全局最优超参数
                is_global_better = False
                if task == 'classification':
                    if final_metrics['acc'] > global_best_val_score:
                        global_best_val_score = final_metrics['acc']
                        is_global_better = True
                else:
                    if final_metrics['rmse'] < global_best_val_score:
                        global_best_val_score = final_metrics['rmse']
                        is_global_better = True

                if is_global_better:
                    global_best_results = final_metrics
                    global_best_params = {"lr": lr, "wd": wd, "actual_epochs": epoch + 1}

    # 格式化输出
    if task == 'classification':
        result_line = f"acc:{global_best_results['acc']:.4f},auc:{global_best_results['auc']:.4f},macro-F1:{global_best_results['f1']:.4f}"
    else:
        result_line = f"rmse:{global_best_results['rmse']:.4f},mae:{global_best_results['mae']:.4f},r2:{global_best_results['r2']:.4f}"

    return result_line, global_best_params

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # --- 1. 路径解析逻辑 (保持不变) ---
    data_root = cfg.get('data_base')
    path_keys = [
        'data_train_eval_imaging', 'labels_train_eval_imaging',
        'data_val_eval_imaging', 'labels_val_eval_imaging',
        'data_test_eval_imaging', 'labels_test_eval_imaging'
    ]
    if data_root:
        for key in path_keys:
            if key in cfg and cfg[key] is not None:
                cfg[key] = os.path.join(data_root, cfg[key])

    # --- 2. 动态生成输出文件名 ---
    target = cfg.get('target', 'default') # 获取 target，若不存在则使用 'default'
    output_filename = f'result/vit_results_{target}.txt'
    
    print(f"\n--- 实验目标: {target} ---")
    print(f"结果将保存至: {output_filename}")

    # 准备数据信息
    train_info = {'paths': cfg.data_train_eval_imaging, 'labels': cfg.labels_train_eval_imaging}
    val_info = {'paths': cfg.data_val_eval_imaging, 'labels': cfg.labels_val_eval_imaging}

    # 3. 运行实验并写入结果
    seeds = [2022, 2023, 2024]
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, 'a') as f:
        f.write(f"--- 最终配置 (Target: {target}) ---\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("-" * 30 + "\n\n")

        for seed in seeds:
            result_line, best_params = run_vit_experiment(cfg, train_info, val_info, seed, target)
            
            f.write(f"seed:{seed}\n")
            f.write(f"best_params: {best_params}\n")
            f.write(result_line + "\n\n")

if __name__ == "__main__":
    main()