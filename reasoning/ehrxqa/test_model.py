import os
import sys

# Importing your specific evaluation modules
from stage1.test_gemini import evaluate_ehrxqa_gemini
from stage1.test_gpt import run_openai_evaluation
from stage1.open_models import evaluate_ehrxqa
from stage2.evaluate_gpt import run_openai_stage2_eval
from stage2.evaluate_gemini import run_gemini_stage2_eval
from stage2.evaluate_open_model import evaluate_hf_vllm

def ehrxqa_benchmark_interface(model_id: str, setting: str):
    """
    Main entry point to route evaluation tasks.
    
    Args:
        model_id (str): The model identifier (e.g., 'gemini', 'gpt-4.1', 
                        or a HF path like 'Qwen/Qwen3-VL-8B-Instruct').
        setting (str): The execution mode ('stage1', 'full', or 'stage2').
    """
    setting = setting.lower()
    mid_lower = model_id.lower()
    
    print(f"[*] Dispatching Task | Setting: {setting} | Model: {model_id}")

    # Logic for Stage 1 and Full Evaluation
    if setting in ["stage1", "full"]:
        if "gemini" in mid_lower:
            print("[INFO] Routing to: Stage 1 Gemini Evaluation")
            return evaluate_ehrxqa_gemini()
        
        elif "gpt-4.1" in mid_lower:
            print("[INFO] Routing to: Stage 1 GPT Evaluation")
            return run_openai_evaluation()
        
        else:
            print(f"[INFO] Routing to: Stage 1 Open Model Evaluation ({model_id})")
            return evaluate_ehrxqa(model_id=model_id)

    # Logic for Stage 2 (Multimodal Reasoning/Visual QA)
    elif setting == "stage2":
        if "gemini" in mid_lower:
            print("[INFO] Routing to: Stage 2 Gemini Evaluation")
            return run_gemini_stage2_eval()
        
        elif "gpt-4.1" in mid_lower:
            print("[INFO] Routing to: Stage 2 GPT Evaluation")
            return run_openai_stage2_eval()
        
        else:
            print(f"[INFO] Routing to: Stage 2 Open Model Evaluation ({model_id})")
            return evaluate_hf_vllm(model_id=model_id)

    else:
        raise ValueError(f"[ERROR] Invalid setting provided: '{setting}'. Choose from 'stage1', 'full', or 'stage2'.")

# Example Usage:
# ehrxqa_benchmark_interface("gemini", "full")
# ehrxqa_benchmark_interface("gpt-4.1", "stage2")
# ehrxqa_benchmark_interface("Qwen/Qwen3-VL-8B-Instruct", "stage1")