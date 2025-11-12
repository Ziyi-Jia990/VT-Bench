CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_breast_TIP.yaml \
    pretrain=False \
    evaluate=True \
    checkpoint=/home/debian/TIP/results/runs/multimodal/breast_cancer_2022/checkpoint_last_epoch_499.ckpt \
    exp_name=finetune_2023_tip \
    max_epochs=500 \
    lr_eval=1e-5 \
    seed=2022


CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_breast_TIP.yaml \
    pretrain=False \
    evaluate=True \
    checkpoint=/home/debian/TIP/results/runs/multimodal/pretrain_2023_breast_cancer_1107_1052/checkpoint_last_epoch_499.ckpt \
    exp_name=finetune_2023_tip \
    max_epochs=500 \
    lr_eval=1e-5 \
    seed=2023

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_breast_TIP.yaml \
    pretrain=False \
    evaluate=True \
    checkpoint=/home/debian/TIP/results/runs/multimodal/tip_2024_breast_cancer/checkpoint_last_epoch_499.ckpt \
    exp_name=finetune_2024_tip \
    max_epochs=500 \
    lr_eval=1e-5 \
    seed=2024