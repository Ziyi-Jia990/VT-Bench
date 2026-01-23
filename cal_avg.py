import os
import pandas as pd

# 根目录
# base_dir = r"Y:\nju-file\tibench_data\ukbiobank\results\runs\eval"
base_dir = '/data1/jiazy/mytip/results/runs/eval'


# ✅ 手动指定需要计算的模型列表（顺序会保留）
model_list = ['MMCL', 'TIP']
# model_list = ['DAFT', 'MAX', 'Concat', 'MUL','image']

# 存放所有模型结果
summary = []

for model_name in model_list:
    model_path = os.path.join(base_dir, model_name)
    if not os.path.isdir(model_path):
        print(f"⚠️ 未找到模型文件夹: {model_path}")
        continue

    # 存放该模型的各年份结果
    model_results = []

    # 遍历年份子文件夹
    for year_folder in os.listdir(model_path):
        folder_path = os.path.join(model_path, year_folder)
        file_path = os.path.join(folder_path, "test_results.csv")

        if os.path.isdir(folder_path) and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                model_results.append(df.iloc[0])
                print(f"✅ {model_name} -> 读取成功: {file_path}")
            except Exception as e:
                print(f"⚠️ {model_name} -> 读取失败: {file_path} -> {e}")

    # 若该模型有结果文件，则计算均值
    if model_results:
        model_df = pd.DataFrame(model_results)
        mean_vals = model_df.mean()
        summary.append({
            "model": model_name,
            "test.acc": mean_vals.get("test.acc", None),
            "test.auc": mean_vals.get("test.auc", None),
            "test.f1": mean_vals.get("test.f1", None),
            "year_count": len(model_results)
        })
    else:
        print(f"❌ {model_name} 未找到任何 test_results.csv")

# 汇总结果
summary_df = pd.DataFrame(summary)
summary_df = summary_df[["model", "year_count", "test.acc", "test.auc", "test.f1"]]

# 输出结果
print("\n=== 指定模型均值汇总 ===")
print(summary_df)

# 保存结果
output_path = os.path.join(base_dir, "selected_models_mean_results.csv")
summary_df.to_csv(output_path, index=False)
print(f"\n📁 汇总结果已保存到: {output_path}")