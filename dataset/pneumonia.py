import pandas as pd
import numpy as np
import os
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

# ================= 配置区域 =================

CSV_PATH = '/mnt/hdd/jiazy/mimic/classification/pneumonia_dataset_100k.csv'
IMAGE_ROOT = '/mnt/hdd/jiazy/mimic/image'
OUTPUT_DIR = '/mnt/hdd/jiazy/mimic/classification/features'

IMG_SIZE = (224, 224) 

TARGET_COL = 'target_pneumonia'
SPLIT_COL = 'split'
CAT_FEATURES = ['gender', 'ViewPosition']
CONT_FEATURES = [
    'age', 'Temperature', 'HeartRate', 'RespRate', 'SpO2', 
    'SysBP', 'WBC', 'Hemoglobin', 'Platelet', 'Glucose', 'Creatinine'
]

# ================= 主处理逻辑 =================

def preprocess_final():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"正在读取 CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # --- Step 1: 基础清洗 ---
    print("Step 1: 清洗数据 (NaN & 路径检查)...")
    check_cols = CAT_FEATURES + CONT_FEATURES + [TARGET_COL, 'image_path', SPLIT_COL]
    df = df.dropna(subset=check_cols)
    
    # 路径检查
    def check_jpg_exists(path_suffix):
        return os.path.exists(os.path.join(IMAGE_ROOT, str(path_suffix)))

    tqdm.pandas(desc="Checking Paths")
    mask_exists = df['image_path'].progress_apply(check_jpg_exists)
    df = df[mask_exists].copy()
    print(f"   -> 清洗后剩余有效样本: {len(df)}")

    # --- Step 2: 特征工程 (严谨模式) ---
    print("Step 2: 处理表格特征 (防止数据泄露)...")
    
    # 2.1 类别编码 (Fit 全量数据是安全的，因为只是建立映射表)
    cat_counts = [] 
    for col in CAT_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        cat_counts.append(len(le.classes_))
        
    # 2.2 连续特征标准化 (!!! 关键修改 !!!)
    # 只使用 'train' 数据来计算均值和方差
    scaler = StandardScaler()
    
    train_mask = df[SPLIT_COL] == 'train'
    if train_mask.sum() == 0:
        raise ValueError("未找到 split='train' 的样本，无法进行标准化拟合！")
        
    # Fit only on TRAIN
    print("   -> 正在根据 Train 集计算均值和方差...")
    scaler.fit(df.loc[train_mask, CONT_FEATURES])
    
    # Transform ALL (Train, Valid, Test)
    df[CONT_FEATURES] = scaler.transform(df[CONT_FEATURES])
    print("   -> 已将标准化应用至所有数据。")

    # 2.3 生成 tabular_lengths
    lengths_list = cat_counts + [1] * len(CONT_FEATURES)
    torch.save(lengths_list, os.path.join(OUTPUT_DIR, 'tabular_lengths.pt'))

    # --- Step 3: 图像转 .npy (带跳过逻辑) ---
    print("Step 3: 转换图像 (Resize -> Save NPY)...")
    
    npy_paths = []
    process_count = 0
    skip_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting Images"):
        orig_path = os.path.join(IMAGE_ROOT, row['image_path'])
        save_path = os.path.splitext(orig_path)[0] + '.npy'
        
        # 记录路径，无论是否新生成的
        npy_paths.append(save_path)
        
        # 如果文件已存在，跳过处理 (节省时间)
        if os.path.exists(save_path):
            skip_count += 1
            continue
            
        try:
            with Image.open(orig_path) as img:
                img = img.convert('RGB')
                img = img.resize(IMG_SIZE, Image.BILINEAR)
                # 保存为 uint8 (0-255) 以节省空间
                img_array = np.array(img) 
                
            np.save(save_path, img_array)
            process_count += 1
            
        except Exception as e:
            print(f"Error processing {orig_path}: {e}")
            # 如果这一张处理失败，把刚刚添加进列表的路径由路径改为 None，以便后续过滤
            npy_paths[-1] = None
            
    df['npy_path'] = npy_paths
    
    # 再次清洗掉转换失败的
    df = df.dropna(subset=['npy_path'])
    print(f"   -> 图像处理结束: 新生成 {process_count} 张, 跳过已存在 {skip_count} 张。")

    # --- Step 4: 保存最终文件 ---
    print("Step 4: 保存 Dataset 文件...")
    
    feature_cols = CAT_FEATURES + CONT_FEATURES
    splits = ['train', 'valid', 'test'] 
    
    for split_name in splits:
        # 处理可能的命名差异
        if split_name == 'valid':
            subset = df[df[SPLIT_COL].isin(['valid', 'validate'])]
        else:
            subset = df[df[SPLIT_COL] == split_name]
            
        if len(subset) == 0:
            continue
            
        print(f"   Saving {split_name}: {len(subset)} samples")
        
        # A. Features CSV
        subset[feature_cols].to_csv(
            os.path.join(OUTPUT_DIR, f'{split_name}_features.csv'),
            header=False, index=False
        )
        
        # B. Labels PT (!!! 修改为 Long 适用于分类 !!!)
        labels_tensor = torch.tensor(subset[TARGET_COL].values, dtype=torch.long)
        torch.save(labels_tensor, os.path.join(OUTPUT_DIR, f'{split_name}_labels.pt'))
        
        # C. Paths PT
        paths_list = subset['npy_path'].tolist()
        torch.save(paths_list, os.path.join(OUTPUT_DIR, f'{split_name}_paths.pt'))

    print("-" * 30)
    print("所有预处理工作已完成。")

if __name__ == "__main__":
    preprocess_final()