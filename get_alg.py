import torch

# checkpoint 文件路径
checkpoint_path = "/home/debian/checkpoint_best_acc.ckpt"

# 加载 checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# 检查是否存在 'algorithm_name' 键
if "algorithm_name" in checkpoint:
    print("algorithm_name:", checkpoint["algorithm_name"])
else:
    print("❌ 'algorithm_name' not found in checkpoint keys.")
    print("Available keys:", checkpoint.keys())

print(checkpoint['hyper_parameters'])