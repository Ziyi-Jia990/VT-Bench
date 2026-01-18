CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_cardiac_Infarction_TIP exp_name=pretrain_Infarction_TIP use_wandb=False max_epochs=1
CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_cardiac_Infarction_MMCL exp_name=pretrain_Infarction_MMCL use_wandb=False max_epochs=1

