CUDA_VISIBLE_DEVICES=1 python -u run.py \
        --config-name config_cardiac_Infarction_TIP \
        pretrain=False \
        evaluate=True \
        checkpoint=/data1/jiazy/mytip/results/runs/multimodal/pretrain_Infarction_TIP_2023_Infarction_0119_0322/checkpoint_last_epoch_499.ckpt \
        exp_name=Infarction_2023 \
        max_epochs=500 \
        lr_eval=1e-3 \
        use_wandb=False \
        seed=2023

CUDA_VISIBLE_DEVICES=1 python -u run.py \
        --config-name config_cardiac_Infarction_TIP \
        pretrain=False \
        evaluate=True \
        checkpoint=/data1/jiazy/mytip/results/runs/multimodal/pretrain_Infarction_TIP_2024_Infarction_0119_1823/checkpoint_last_epoch_499.ckpt \
        exp_name=Infarction_2024 \
        max_epochs=500 \
        lr_eval=1e-3 \
        use_wandb=False \
        seed=2024