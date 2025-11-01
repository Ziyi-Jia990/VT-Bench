#!/bin/bash

#   - 3.e-2
#   - 1.e-2
#   - 3.e-3
#   - 1.e-3
#   - 3.e-4
#   - 1.e-4


# --- 定义要搜索的超参数 ---
LEARNING_RATES=(3e-2 1e-2 3e-3 1e-3 3e-4 1e-4)

# --- 用于记录失败的组合 ---
FAILED_RUNS=()
SUCCESSFUL_RUNS=0
TOTAL_RUNS=0

# --- 嵌套循环，遍历所有组合 ---

for lr in "${LEARNING_RATES[@]}"; do

# 增加总运行次数计数器
((TOTAL_RUNS++))

# 打印当前正在运行的组合，方便跟踪
echo "======================================================"
echo "RUNNING: Learning Rate = $lr"
echo "======================================================"

# 执行您的训练脚本，并通过命令行覆盖config.yaml中的参数
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_adoption_DAFT exp_name=pretrain  lr_classifier=$lr

# --- 关键改动：检查上一条命令的退出码 ---
# $? 存储了上一条命令的退出码。0代表成功，非0代表失败。
if [ $? -ne 0 ]; then
    # 如果失败了
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! FAILED: Batch Size = $bs, Learning Rate = $lr"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    # 将失败的组合记录到数组中
    FAILED_RUNS+=("Batch Size=$bs, Learning Rate=$lr")
else
    # 如果成功了
    echo "------------------------------------------------------"
    echo "--- SUCCESS: Batch Size = $bs, Learning Rate = $lr"
    echo "------------------------------------------------------"
    ((SUCCESSFUL_RUNS++))
fi

# 添加一些间隔，让日志更清晰
echo -e "\n"

done


# --- 脚本结束时打印总结报告 ---
echo "===================== GRID SEARCH SUMMARY ====================="
echo "Total runs: $TOTAL_RUNS"
echo "Successful runs: $SUCCESSFUL_RUNS"
echo "Failed runs: $((${#FAILED_RUNS[@]}))"

if [ ${#FAILED_RUNS[@]} -ne 0 ]; then
  echo "-------------------------------------------------------------"
  echo "The following combinations failed:"
  for run in "${FAILED_RUNS[@]}"; do
    echo "  - $run"
  done
  echo "-------------------------------------------------------------"
fi

echo "Grid search finished."