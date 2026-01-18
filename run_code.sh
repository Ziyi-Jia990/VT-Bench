CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_MUL.yaml exp_name=pneumonia_MUL_2023 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_MUL.yaml exp_name=pneumonia_MUL_2024 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2024 

# MAX
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_MAX.yaml exp_name=pneumonia_MAX_2023 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_MAX.yaml exp_name=pneumonia_MAX_2024 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2024 

# Concat
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_Concat.yaml exp_name=pneumonia_Concat_2023 max_epochs=500 \
    lr_eval=1e-5 \
    lr=1e-5 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_Concat.yaml exp_name=pneumonia_Concat_2024 max_epochs=500 \
    lr_eval=1e-5 \
    lr=1e-5 \
    use_wandb=False \
    seed=2024 

# DAFT
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_DAFT.yaml exp_name=pneumonia_DAFT_2023 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_DAFT.yaml exp_name=pneumonia_DAFT_2024 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2024 

# image
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_image.yaml exp_name=pneumonia_image_2023 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_image.yaml exp_name=pneumonia_image_2024 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2024 


# ===  los  ====
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_MUL.yaml exp_name=los_MUL_2023 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_MUL.yaml exp_name=los_MUL_2024 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2024 

# MAX
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_MAX.yaml exp_name=los_MAX_2023 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_MAX.yaml exp_name=los_MAX_2024 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2024 

# Concat
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_Concat.yaml exp_name=los_Concat_2023 max_epochs=500 \
    lr_eval=1e-4 \
    lr=1e-4 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_Concat.yaml exp_name=los_Concat_2024 max_epochs=500 \
    lr_eval=1e-4 \
    lr=1e-4 \
    use_wandb=False \
    seed=2024 

# DAFT
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_DAFT.yaml exp_name=los_DAFT_2023 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_DAFT.yaml exp_name=los_DAFT_2024 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2024 

# image
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_image.yaml exp_name=los_image_2023 max_epochs=500 \
    lr_eval=1e-4 \
    lr=1e-4 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_image.yaml exp_name=los_image_2024 max_epochs=500 \
    lr_eval=1e-4 \
    lr=1e-4 \
    use_wandb=False \
    seed=2024 

# ==== rr ==== 
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_MUL.yaml exp_name=rr_MUL_2023 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_MUL.yaml exp_name=rr_MUL_2024 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2024 

# MAX
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_MAX.yaml exp_name=rr_MAX_2023 max_epochs=500 \
    lr_eval=1e-4 \
    lr=1e-4 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_MAX.yaml exp_name=rr_MAX_2024 max_epochs=500 \
    lr_eval=1e-4 \
    lr=1e-4 \
    use_wandb=False \
    seed=2024 

# Concat
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_Concat.yaml exp_name=rr_Concat_2023 max_epochs=500 \
    lr_eval=1e-4 \
    lr=1e-4 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_Concat.yaml exp_name=rr_Concat_2024 max_epochs=500 \
    lr_eval=1e-4 \
    lr=1e-4 \
    use_wandb=False \
    seed=2024 

# DAFT
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_DAFT.yaml exp_name=rr_DAFT_2023 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_DAFT.yaml exp_name=rr_DAFT_2024 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2024 

# image
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_image.yaml exp_name=rr_image_2023 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2023 

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_image.yaml exp_name=rr_image_2024 max_epochs=500 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False \
    seed=2024 

# scp -r /home/jiazy/mytip/results/runs/eval/los_MUL_*_los_* pi@10.50.16.39:/home/pi/shared/nju-file/tibench_data/los/MUL
# MAX Concat DAFT image