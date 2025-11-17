CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_breast_image.yaml \
    exp_name=finetune_2022_image \
    max_epochs=500 \
    seed=2022


CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_breast_image.yaml \
    exp_name=finetune_2023_image \
    max_epochs=500 \
    seed=2023

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_breast_image.yaml \
    exp_name=finetune_2024_image \
    max_epochs=500 \
    seed=2024