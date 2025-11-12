# import torch

# # checkpoint 文件路径
# checkpoint_path = "/home/debian/TIP/results/runs/multimodal/pretrain_2024_dvm_1031_1449/checkpoint_last_epoch_499.ckpt"

# # 加载 checkpoint
# checkpoint = torch.load(checkpoint_path, map_location="cpu")

# # 检查是否存在 'algorithm_name' 键
# if "algorithm_name" in checkpoint:
#     print("algorithm_name:", checkpoint["algorithm_name"])
# else:
#     print("❌ 'algorithm_name' not found in checkpoint keys.")
#     print("Available keys:", checkpoint.keys())

# print(checkpoint['hyper_parameters']['seed'])

import torch
from omegaconf import OmegaConf, open_dict

# --- 1. 请修改这里的路径 ---
CHECKPOINT_PATH_IN = '/home/debian/TIP/results/runs/multimodal/pretrain_2023_breast_cancer_1107_1052/checkpoint_last_epoch_499_raw.ckpt' # 替换成你的原始 checkpoint 路径
CHECKPOINT_PATH_OUT = "/home/debian/TIP/results/runs/multimodal/pretrain_2023_breast_cancer_1107_1052/checkpoint_last_epoch_499.ckpt"  # 修复后的新路径

# --- 2. 请定义正确的值 ---
# 从你的 DEBUG 输出中，我们知道正确的 num_cat 是 5
NEW_NUM_CAT = 5

# !!! 关键：请在这里填入你正确的 "连续特征" 数量 !!!
# 你需要确认你的 field_lengths_tabular 文件中有多少个连续特征（值为 1）
# 假设你的描述 "1个连续特征" 是对的，那么这里就填 1
NEW_NUM_CON = 1  # <--- 请根据你的数据确认并修改这个值！

# --------------------------

print(f"正在加载 checkpoint: {CHECKPOINT_PATH_IN}")
# 加载到 cpu，避免占用 GPU
checkpoint = torch.load(CHECKPOINT_PATH_IN, map_location='cpu')

# 访问 hparams (它是一个 OmegaConf 对象)
hparams = checkpoint['hyper_parameters']
print("------------------------------------------")
print(f"原始 hparams: num_cat={hparams.get('num_cat')}, num_con={hparams.get('num_con')}")

# 使用 open_dict 来允许修改 OmegaConf
try:
    with open_dict(hparams):
        hparams.num_cat = NEW_NUM_CAT
        hparams.num_con = NEW_NUM_CON
except Exception as e:
    print(f"修改 hparams 出错: {e}")
    print("请检查你的 checkpoint 结构是否正确。")
    exit()

print(f"修改后 hparams: num_cat={hparams.num_cat}, num_con={hparams.num_con}")
print("------------------------------------------")

# 4. 保存新的 (修复后的) checkpoint
try:
    torch.save(checkpoint, CHECKPOINT_PATH_OUT)
    print(f"已成功保存修复后的文件到: {CHECKPOINT_PATH_OUT}")
    print("\n现在请在你的评估命令中改用这个新的 .ckpt 文件。")
except Exception as e:
    print(f"保存新文件时出错: {e}")