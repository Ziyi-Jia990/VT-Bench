import os
import sys

def evaluate_model(model_id, **kwargs):
    """
    Orchestrator function that routes the evaluation task to the correct 
    model-specific script based on the model_id.
    """
    
    model_id_lower = model_id.lower()

    # 1. Qwen3 Family Routing (Thinking & Instruct)
    if "qwen3-vl" in model_id_lower:
        from .models.test_qwen_vl import run_qwen3_evaluation
        return run_qwen3_evaluation(
            model_id=model_id
        )

    # 2. InternVL Family
    elif "internvl" in model_id_lower:
        from .models.InternVL3 import run_internvl
        return run_internvl(
            model_id=model_id
        )

    # 3. Pixtral Family
    elif "pixtral" in model_id_lower:
        from .models.pixtral import run_pixtral_evaluation
        return run_pixtral_evaluation(
            model_id=model_id
        )

    # 4. GLM Family (Local Thinking Models)
    elif "glm" in model_id_lower:
        from .models.test_glm import run_glm_evaluation
        return run_glm_evaluation(
            model_id=model_id,
            cache_dir=kwargs.get("cache_dir")
        )

    # 5. Llama Family
    elif "llama" in model_id_lower:
        from .models.test_llama3 import run_llama_evaluation
        return run_llama_evaluation(
            model_id=model_id
        )

    # 6. Gemini Family (API)
    elif "gemini" in model_id_lower:
        from .models.test_gemini import run_gemini_mmqa_evaluation
        return run_gemini_mmqa_evaluation(
            api_key=kwargs.get("api_key"),
            model_name=model_id,
            rpm=kwargs.get("rpm", 10),
            tpm=kwargs.get("tpm", 1000000)
        )

    # 7. GPT Family (API)
    elif "gpt" in model_id_lower:
        from .models.test_gpt4 import run_gpt_evaluation
        return run_gpt_evaluation(
            api_key=kwargs.get("api_key"),
            model_name=model_id,
            rpm=kwargs.get("rpm", 3),
            tpm=kwargs.get("tpm", 50000)
        )

    else:
        print(f" Error: Model ID '{model_id}' does not match any known routing rules.")
        return None