import argparse
import sys

# Import required modules and functions
try:
    from reasoning.run_reasoning import reasoning_evaluation
    from prediction.run import call_with_specific_config
    from prediction.MCR_calculate import load_and_run_from_checkpoint
except ImportError as e:
    print(f"Error: Could not import modules. Please check your file paths. Details: {e}")
    sys.exit(1)

def main():
    # 1. Initialize the argument parser
    parser = argparse.ArgumentParser(description="Multi-task script for Prediction and Reasoning evaluation.")
    
    # 2. Define command-line arguments
    parser.add_argument("--task", type=str, required=True, help="Task type: 'prediction' or 'reasoning'")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--model", type=str, required=True, help="Model architecture/name")
    parser.add_argument("--setting", type=str, required=True, help="Experimental setting")
    parser.add_argument("--diagnostics", type=str, required=True, help="Diagnostic mode: 'full', 'mcr', or others")
    parser.add_argument("--checkpoints", type=str, default="", help="Path to a manual checkpoint (optional)")

    args = parser.parse_args()

    # 3. Execution logic based on task type
    if args.task == "prediction":
        # Generate config filename based on dataset and model
        config_name = f"config_{args.dataset}_{args.model}.yaml"
        print(f"[*] Running prediction using config: {config_name}")
        
        # Execute prediction and retrieve the trained checkpoint path
        checkpoints_train = call_with_specific_config(
            config_name,
            model_id=args.model,
            diagnosis=args.diagnostics,
        )

        # Logic for diagnostics (full or mcr)
        if args.diagnostics in ["full", "mcr"]:
            # Check if a manual checkpoint was provided via command line
            if args.checkpoints and args.checkpoints.strip():
                print(f"[*] Diagnostics triggered. Loading provided checkpoint: {args.checkpoints}")
                load_and_run_from_checkpoint(args.checkpoints)
            else:
                # Fallback to the checkpoint returned by the training process
                print(f"[*] Diagnostics triggered. Loading trained checkpoint: {checkpoints_train}")
                load_and_run_from_checkpoint(checkpoints_train)

    elif args.task == "reasoning":
        # Execute reasoning evaluation logic
        print(f"[*] Starting reasoning evaluation for model: {args.model} on dataset: {args.dataset}")
        reasoning_evaluation(args.model, args.dataset, args.setting)
    
    else:
        # Handle unsupported task types
        print(f"Error: Unknown task type '{args.task}'")
        sys.exit(1)

if __name__ == "__main__":
    main()