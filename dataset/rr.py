import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from PIL import Image
from tqdm import tqdm

# 确保 PIL 版本兼容
try:
    LANCZOS_RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS_RESAMPLE = Image.LANCZOS

# --- 1. 定义常量和路径 ---
# [配置] 输入路径 (RR 数据集)
CSV_FILE = "/mnt/hdd/jiazy/mimic/regression/rr/rr_dataset_100k.csv"
# 图片根目录保持不变
IMAGE_ROOT = "/mnt/hdd/jiazy/mimic/image"

# [配置] 输出目录 (RR 特征输出)
OUTPUT_DIR = "/mnt/hdd/jiazy/mimic/regression/rr/features"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. 定义特征列 ---
# 连续特征 (13个, 已移除 RespRate)
CONTINUOUS_COLS = [
    'age', 'Temperature', 'HeartRate', 'SpO2', 'SysBP', 'DiaBP', 
    'WBC', 'Hemoglobin', 'Platelet', 'Glucose', 'Creatinine', 
    'Sodium', 'Potassium'
]

# 类别特征 (2个)
CATEGORICAL_COLS = [
    'gender', 'ViewPosition'
]

# 标签列 (呼吸率数值)
LABEL_COL = 'target_rr_value'

# 需要使用的列
USE_COLS = ['split', 'image_path', LABEL_COL] + CONTINUOUS_COLS + CATEGORICAL_COLS

# --- 3. 图像处理辅助函数 ---
def process_and_save_image(rel_path):
    """
    加载图像, resize 到 224x224 (不裁剪), 并另存为 .npy 文件。
    """
    if pd.isna(rel_path):
        return None

    # 拼接完整路径
    full_img_path = os.path.join(IMAGE_ROOT, rel_path)
    
    # 构造 .npy 保存路径
    npy_path = full_img_path.replace(".jpg", ".npy").replace(".jpeg", ".npy").replace(".png", ".npy")

    # 如果 .npy 已经存在，直接返回路径
    if os.path.exists(npy_path):
        return npy_path

    if not os.path.exists(full_img_path):
        return None 

    try:
        img = Image.open(full_img_path)
        img_resized = img.resize((224, 224), resample=LANCZOS_RESAMPLE)
        
        # 转换为 RGB
        if img_resized.mode != 'RGB':
            img_resized = img_resized.convert('RGB')
            
        np_img = np.array(img_resized)
        np.save(npy_path, np_img)
        
        return npy_path
    except Exception as e:
        print(f"处理图像失败 {full_img_path}: {e}")
        return None

def preprocess_mimic_rr_flow():
    """
    预处理 MIMIC-CXR (RR Regression) 数据集
    """
    print("开始执行 MIMIC-CXR (RR Regression) 预处理流程...")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # --- 1. 加载数据 ---
    print(f"正在加载元数据 {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE, usecols=USE_COLS)
    print(f"原始数据条数: {len(df)}")

    # --- 2. 基础数据清洗 ---
    df = df.dropna(subset=[LABEL_COL])
    print(f"过滤无标签数据后条数: {len(df)}")
    
    # --- 3. 类别特征编码 ---
    print("正在进行类别特征编码...")
    cat_dims = []
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(-1) 
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
        num_cats = int(df[col].max() + 1)
        cat_dims.append(num_cats if num_cats > 0 else 1)
    
    print(f"类别特征维度: {dict(zip(CATEGORICAL_COLS, cat_dims))}")

    # --- 4. 数据集划分 ---
    print("正在基于 'split' 列进行数据集划分...")
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'valid'].copy()
    test_df = df[df['split'] == 'test'].copy()

    print(f"划分结果: 训练集 {len(train_df)}, 验证集 {len(val_df)}, 测试集 {len(test_df)}")

    # --- 5. 连续特征标准化 ---
    print("正在标准化连续特征...")
    if CONTINUOUS_COLS:
        scaler = StandardScaler()
        # 填充缺失值
        train_df[CONTINUOUS_COLS] = train_df[CONTINUOUS_COLS].fillna(0)
        val_df[CONTINUOUS_COLS] = val_df[CONTINUOUS_COLS].fillna(0)
        test_df[CONTINUOUS_COLS] = test_df[CONTINUOUS_COLS].fillna(0)

        scaler.fit(train_df[CONTINUOUS_COLS])
        
        train_df[CONTINUOUS_COLS] = scaler.transform(train_df[CONTINUOUS_COLS])
        val_df[CONTINUOUS_COLS] = scaler.transform(val_df[CONTINUOUS_COLS])
        test_df[CONTINUOUS_COLS] = scaler.transform(test_df[CONTINUOUS_COLS])

    # --- 6. 图像处理 ---
    print("正在处理图像 (Resize & Save .npy)...")
    tqdm.pandas(desc="Train Images")
    train_df['npy_path'] = train_df['image_path'].progress_apply(process_and_save_image)
    
    tqdm.pandas(desc="Val Images")
    val_df['npy_path'] = val_df['image_path'].progress_apply(process_and_save_image)
    
    tqdm.pandas(desc="Test Images")
    test_df['npy_path'] = test_df['image_path'].progress_apply(process_and_save_image)

    # 过滤无效图像
    len_before = len(train_df) + len(val_df) + len(test_df)
    train_df = train_df.dropna(subset=['npy_path'])
    val_df = val_df.dropna(subset=['npy_path'])
    test_df = test_df.dropna(subset=['npy_path'])
    len_after = len(train_df) + len(val_df) + len(test_df)
    
    if len_before != len_after:
        print(f"⚠️ 警告：过滤了 {len_before - len_after} 条图像不存在或损坏的数据。")

    # --- 7. 保存输出 ---
    print(f"[Final] 正在保存处理后的文件到 {OUTPUT_DIR} ...")

    # 保存 tabular_lengths
    tabular_lengths = cat_dims + [1] * len(CONTINUOUS_COLS)
    torch.save(tabular_lengths, os.path.join(OUTPUT_DIR, "tabular_lengths.pt"))
    print(f"特征维度保存完毕: 类别{len(cat_dims)} + 连续{len(CONTINUOUS_COLS)}")
    
    for split_name, df_split in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        # 1. 保存特征 CSV
        features_path = os.path.join(OUTPUT_DIR, f"{split_name}_features.csv")
        cols_to_save = CATEGORICAL_COLS + CONTINUOUS_COLS
        df_split[cols_to_save].to_csv(features_path, index=False, header=False)
        
        # 2. 保存标签 Tensor
        labels_path = os.path.join(OUTPUT_DIR, f"{split_name}_labels.pt")
        labels_tensor = torch.tensor(df_split[LABEL_COL].values, dtype=torch.float32)
        torch.save(labels_tensor, labels_path)
        
        # 3. 保存路径 List
        paths_path = os.path.join(OUTPUT_DIR, f"{split_name}_paths.pt")
        npy_path_list = df_split['npy_path'].tolist()
        torch.save(npy_path_list, paths_path)

    print("✅ MIMIC (RR) 预处理全部完成！")

if __name__ == "__main__":
    preprocess_mimic_rr_flow()