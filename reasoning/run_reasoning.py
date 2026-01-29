import os
import sys

def reasoning_evaluation(model_id: str, dataset: str, setting):
    """
    Central dispatcher to route evaluation based on the chosen dataset.
    
    Args:
        model_id (str): The identifier of the model to evaluate.
        dataset (str): The target dataset name ('dvm', 'ehrxqa', etc.).
        setting (str or list): The evaluation mode. Will be normalized to a list.
    """
    dataset_type = dataset.lower().strip()
    
    # --- 统一处理 setting 为 list ---
    if isinstance(setting, str):
        # 如果传的是字符串（如 "stage1"），转为 ["stage1"]
        setting_list = [setting]
    elif isinstance(setting, list):
        setting_list = setting
    else:
        # 容错处理：转为列表
        setting_list = list(setting)

    print(f"[*] Global Dispatcher | Dataset: {dataset_type} | Model: {model_id} | Setting: {setting_list}")

    # 1. Route to DVM Dataset Logic
    if dataset_type == "dvm":
        print(f"[INFO] Routing to DVM Evaluation...")
        from .reasoning.DVM_QA.test_model import evaluate_dvm
        # DVM 接收整个 list
        return evaluate_dvm(model_id=model_id, tasks=setting_list)

    # 2. Route to EHRXQA Dataset Logic
    elif dataset_type == "ehrxqa":
        print(f"[INFO] Routing to EHRXQA Benchmark Interface...")
        from .reasoning.ehrxqa.test_model import ehrxqa_benchmark_interface
        # EHRXQA 取 list 的第一个参数 (str)
        actual_setting = setting_list[0] if setting_list else "full"
        return ehrxqa_benchmark_interface(model_id=model_id, setting=actual_setting)

    # 3. Fallback for other datasets (e.g., MultimodalQA / mmqa)
    else:
        print(f"[INFO] Dataset '{dataset}' not specifically routed. Calling generic evaluate_model...")
        from .reasoning.mmqa.test_models import evaluate_model
        return evaluate_model(model_id=model_id)