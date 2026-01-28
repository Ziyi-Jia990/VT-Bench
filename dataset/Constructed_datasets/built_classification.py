import pandas as pd
import numpy as np
import os
from tqdm import tqdm
# ================= 配置区域 =================
CONFIG = {
    "MIMIC_IV_HOSP": "../physionet.org/files/mimiciv/2.2/hosp",
    "MIMIC_IV_ICU": "../physionet.org/files/mimiciv/2.2/icu",
    "MIMIC_CXR": "../physionet.org/files/mimic-cxr-jpg/2.0.0",
    "OUTPUT_DIR": "/mnt/hdd/jiazy/ehrxqa/classification",
    "TARGET_COUNT": 100000,
}

# 定义我们要提取的特征 (ItemID 字典)
# 这些是 MIMIC-IV 中最常用的 ItemID
FEATURES_CONFIG = {
    'chartevents': {
        'Temperature': [223761, 223762], # F and C
        'HeartRate': [220045],
        'RespRate': [220210],
        'SpO2': [220277],
        'SysBP': [220179, 220050], # 收缩压 (非侵入/侵入)
    },
    'labevents': {
        'WBC': [51301, 51300], # 白细胞
        'Hemoglobin': [51222], # 血红蛋白
        'Platelet': [51265],   # 血小板
        'Glucose': [50931],    # 血糖
        'Creatinine': [50912]  # 肌酐 (肾功能)
    }
}
# ===========================================

def get_file_path(base_dir, filename):
    path_gz = os.path.join(base_dir, filename + ".gz")
    path_csv = os.path.join(base_dir, filename)
    if os.path.exists(path_gz): return path_gz
    if os.path.exists(path_csv): return path_csv
    raise FileNotFoundError(f"Cannot find {filename} in {base_dir}")

def generate_image_path(row):
    subject_id = str(int(row['subject_id']))
    study_id = str(int(row['study_id']))
    image_id = str(row['image_id'])
    return f"p{subject_id[:2]}/p{subject_id}/s{study_id}/{image_id}.jpg"

def extract_clinical_features(df_cohort, lookback_hours=24):
    """
    通用函数：从 ICU 和 Hosp 表中提取临床特征 (兼容旧版 Pandas)
    """
    print(f"    Extracting Clinical Features (Window: +/- {lookback_hours} hours)...")
    valid_hadms = set(df_cohort['hadm_id'].unique())
    
    # 收集所有的 ItemID 以便过滤
    target_chart_ids = [item for sublist in FEATURES_CONFIG['chartevents'].values() for item in sublist]
    target_lab_ids = [item for sublist in FEATURES_CONFIG['labevents'].values() for item in sublist]
    
    extracted_data = []

    # --- 1.处理 Vitals (ChartEvents) ---
    print("    Scanning Vital Signs (Chartevents)...")
    chart_path = get_file_path(CONFIG["MIMIC_IV_ICU"], "chartevents.csv")
    chunk_size = 5000000
    
    # 建立 ID 到名称的映射
    item_map = {}
    for name, ids in FEATURES_CONFIG['chartevents'].items():
        for i in ids: item_map[i] = name

    # 【修改点 1】去掉 with，直接赋值
    reader = pd.read_csv(chart_path, chunksize=chunk_size, usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'])
    
    for chunk in tqdm(reader, desc="    Reading Vitals"):
        # 过滤
        chunk = chunk[chunk['hadm_id'].isin(valid_hadms)]
        chunk = chunk[chunk['itemid'].isin(target_chart_ids)]
        chunk = chunk.dropna(subset=['valuenum'])
        
        # 温度转换 (华氏度 223761 -> 摄氏度)
        if 223761 in chunk['itemid'].values:
            f_mask = chunk['itemid'] == 223761
            chunk.loc[f_mask, 'valuenum'] = (chunk.loc[f_mask, 'valuenum'] - 32) * 5/9
            chunk.loc[f_mask, 'itemid'] = 223762 # 映射到摄氏度ID
        
        if not chunk.empty:
            extracted_data.append(chunk)

    # --- 2.处理 Labs (LabEvents) ---
    print("    Scanning Lab Tests (Labevents)...")
    lab_path = get_file_path(CONFIG["MIMIC_IV_HOSP"], "labevents.csv")
    
    # 更新映射
    for name, ids in FEATURES_CONFIG['labevents'].items():
        for i in ids: item_map[i] = name
        
    # 【修改点 2】去掉 with，直接赋值
    reader_lab = pd.read_csv(lab_path, chunksize=chunk_size, usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'])
    
    for chunk in tqdm(reader_lab, desc="    Reading Labs"):
        chunk = chunk[chunk['hadm_id'].isin(valid_hadms)]
        chunk = chunk[chunk['itemid'].isin(target_lab_ids)]
        chunk = chunk.dropna(subset=['valuenum'])
        
        if not chunk.empty:
            extracted_data.append(chunk)
    
    if not extracted_data:
        print("    Warning: No clinical features found.")
        return df_cohort

    # --- 3. 合并与时间窗口过滤 ---
    print("    Merging and Aggregating Features...")
    df_features = pd.concat(extracted_data)
    df_features['charttime'] = pd.to_datetime(df_features['charttime'])
    
    # 将特征名称映射上去
    df_features['feature_name'] = df_features['itemid'].map(item_map)
    
    # 连接到 Cohort (为了获取 studydatetime)
    # 优化内存：只取必要的列
    cohort_time = df_cohort[['hadm_id', 'study_id', 'studydatetime']].drop_duplicates()
    merged = pd.merge(df_features, cohort_time, on='hadm_id', how='inner')
    
    # 计算时间差
    merged['hours_diff'] = (merged['charttime'] - merged['studydatetime']).dt.total_seconds() / 3600
    
    # 过滤时间窗口
    merged = merged[abs(merged['hours_diff']) <= lookback_hours]
    
    # --- 4. 透视表 (Pivoting) ---
    # 取平均值作为特征
    df_pivot = pd.pivot_table(merged, values='valuenum', index='study_id', columns='feature_name', aggfunc='mean')
    
    # 合并回原始数据
    df_final = pd.merge(df_cohort, df_pivot, on='study_id', how='left')
    return df_final


def split_data(df):
    """8:1:1 划分"""
    print("    Splitting data into train (80%), valid (10%), test (10%)...")
    subjects = df['subject_id'].unique()
    np.random.seed(42)
    np.random.shuffle(subjects)
    
    n = len(subjects)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    
    train_subj = set(subjects[:n_train])
    valid_subj = set(subjects[n_train:n_train+n_valid])
    
    def get_split(sid):
        if sid in train_subj: return 'train'
        if sid in valid_subj: return 'valid'
        return 'test'
        
    df['split'] = df['subject_id'].apply(get_split)
    print("    Split stats:")
    print(df['split'].value_counts())
    return df

def step_1_link_cxr_to_hadm():
    """
    Step 1: 全量对齐
    不再限制 limit_count，尽可能获取所有可用的图片-住院匹配对
    """
    print(">>> Step 1: Loading Metadata and Aligning CXR to Admissions (Full Scale)...")
    
    cxr_meta_path = get_file_path(CONFIG["MIMIC_CXR"], "mimic-cxr-2.0.0-metadata.csv")
    
    # 读取所有数据
    df_cxr = pd.read_csv(cxr_meta_path, usecols=['subject_id', 'study_id', 'dicom_id', 'StudyDate', 'StudyTime', 'ViewPosition'])
    df_cxr.rename(columns={'dicom_id': 'image_id'}, inplace=True)
    
    # 过滤 ViewPosition
    df_cxr = df_cxr[df_cxr['ViewPosition'].isin(['AP', 'PA'])]

    # 【关键修改】不再截断数据，利用你的大内存处理全量数据
    # df_cxr = df_cxr.iloc[:limit_count]  <-- 这一行删掉或注释掉
    print(f"    Loaded {len(df_cxr)} CXR studies candidates.")

    # 处理时间
    df_cxr['studydatetime'] = pd.to_datetime(
        df_cxr['StudyDate'].astype(str) + ' ' + 
        df_cxr['StudyTime'].astype(str).str.split('.').str[0].str.zfill(6),
        format='%Y%m%d %H%M%S', errors='coerce'
    )
    
    # 读取 Transfers 对齐
    trans_path = get_file_path(CONFIG["MIMIC_IV_HOSP"], "transfers.csv")
    df_trans = pd.read_csv(trans_path, usecols=['subject_id', 'hadm_id', 'intime', 'outtime'])
    df_trans = df_trans.dropna(subset=['hadm_id', 'intime', 'outtime'])
    df_trans['intime'] = pd.to_datetime(df_trans['intime'])
    df_trans['outtime'] = pd.to_datetime(df_trans['outtime'])
    
    valid_subjects = set(df_cxr['subject_id'].unique())
    df_trans = df_trans[df_trans['subject_id'].isin(valid_subjects)]
    
    print("    Merging for alignment...")
    merged = pd.merge(df_cxr, df_trans, on='subject_id', how='left')
    mask = (merged['studydatetime'] >= merged['intime']) & (merged['studydatetime'] <= merged['outtime'])
    aligned = merged[mask].copy()
    aligned = aligned.drop_duplicates(subset=['study_id', 'image_id'])
    
    print(f"    Aligned {len(aligned)} images (Full Set).")
    return aligned[['subject_id', 'study_id', 'image_id', 'hadm_id', 'studydatetime', 'ViewPosition']]


def step_3_build_pneumonia_dataset(df_aligned):
    """
    任务 2: 肺炎预测 (严格过滤缺失值版)
    """
    print("\n>>> Step 3: Building Pneumonia Dataset (Strict filtering)...")
    
    # 1. 确定 Label
    diag_path = get_file_path(CONFIG["MIMIC_IV_HOSP"], "diagnoses_icd.csv")
    valid_hadms = set(df_aligned['hadm_id'].unique())
    pneumonia_hadms = set()
    pneu_prefixes = tuple(['480', '481', '482', '483', '484', '485', '486', '487', 
                           'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18'])
    
    chunk_size = 5000000
    reader = pd.read_csv(diag_path, chunksize=chunk_size, usecols=['hadm_id', 'icd_code'])
    for chunk in tqdm(reader, desc="    Scanning Diagnoses"):
        chunk = chunk[chunk['hadm_id'].isin(valid_hadms)].copy()
        chunk['icd_code'] = chunk['icd_code'].astype(str)
        pneu_chunk = chunk[chunk['icd_code'].str.startswith(pneu_prefixes)]
        if not pneu_chunk.empty:
            pneumonia_hadms.update(pneu_chunk['hadm_id'].unique())
            
    df_aligned['target_pneumonia'] = df_aligned['hadm_id'].apply(lambda x: 1 if x in pneumonia_hadms else 0)
    
    # 2. 补充基础人口学特征
    pat_path = get_file_path(CONFIG["MIMIC_IV_HOSP"], "patients.csv")
    df_pat = pd.read_csv(pat_path, usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year'])
    df_final = pd.merge(df_aligned, df_pat, on='subject_id', how='inner')
    df_final['admit_year'] = df_final['studydatetime'].dt.year
    df_final['age'] = df_final['anchor_age'] + (df_final['admit_year'] - df_final['anchor_year'])
    
    # 【关键修改 1】移除此处的截断逻辑
    # 我们要对所有对齐的数据进行特征提取，这样才能保证筛选后还有足够的数据
    print(f"    Processing all {len(df_final)} candidates for feature extraction...")
    
    # 3. 提取丰富的临床特征 (全量)
    # 这会消耗较多内存，但你的 64G 内存完全足以应付
    df_final = extract_clinical_features(df_final, lookback_hours=24)
    
    # 4. 【关键修改 2】严格过滤缺失值
    print("    Filtering out rows with missing clinical features...")
    
    # 获取所有特征列名
    feature_cols = list(FEATURES_CONFIG['chartevents'].keys()) + list(FEATURES_CONFIG['labevents'].keys())
    # 确保列都在 dataframe 里
    feature_cols = [c for c in feature_cols if c in df_final.columns]
    
    initial_len = len(df_final)
    # 只要有任何一个特征缺失 (NaN)，就整行删除
    df_final = df_final.dropna(subset=feature_cols, how='any')
    
    print(f"    Dropped {initial_len - len(df_final)} rows with missing values.")
    print(f"    Remaining clean samples: {len(df_final)}")
    
    # 5. 【关键修改 3】后置截断
    # 只有当清洗后的数据量超过目标值时，才进行随机采样
    if len(df_final) > CONFIG["TARGET_COUNT"]:
        print(f"    Limiting dataset to target count: {CONFIG['TARGET_COUNT']}...")
        df_final = df_final.sample(n=CONFIG["TARGET_COUNT"], random_state=42).reset_index(drop=True)
    elif len(df_final) < CONFIG["TARGET_COUNT"]:
        print(f"    Warning: Only found {len(df_final)} valid samples (Target: {CONFIG['TARGET_COUNT']}).")
        print("    Tips: You can try increasing lookback_hours in extract_clinical_features.")

    # 6. 生成路径和分割
    df_final['image_path'] = df_final.apply(generate_image_path, axis=1)
    df_final = split_data(df_final)
    
    cols = ['split', 'subject_id', 'study_id', 'image_path', 'gender', 'age', 'ViewPosition', 'target_pneumonia'] + feature_cols
    
    save_path = os.path.join(CONFIG["OUTPUT_DIR"], "pneumonia_dataset_100k_enriched.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_final[cols].to_csv(save_path, index=False)
    
    print(f"    Positive samples: {df_final['target_pneumonia'].sum()}")
    print(f"    Final Dataset saved to: {save_path}")





# 请确保将上面的 extract_clinical_features 和 step_3 替换进原本的 build_classification.py 中
# 别忘了运行 step_1_link_cxr_to_hadm 获取 df_aligned
if __name__ == "__main__":
    df_aligned = step_1_link_cxr_to_hadm()
    # step_2_build_mortality_dataset(df_aligned)
    step_3_build_pneumonia_dataset(df_aligned)
    print("\nClassification dataset construction complete!")