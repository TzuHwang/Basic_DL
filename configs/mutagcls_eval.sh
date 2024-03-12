set -e
set -x

data_root="D:/data/MUTAG"
output_root="./outputs"
config_name="mutagcls"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --data-root ${data_root} \
    --output-root ${output_root} \
    --config-name ${config_name} \
    \
    --section final_val \
    \
    --dataset MUTAG \
    --data-format graph \
    --aug default \
    --crop 0.8 \
    --loading-method all \
    --normalize \
    --maxv 255.0 \
    --batch-size 1 \
    --num-workers 4 \
    --no-label-data-portion 0.8 \
    \
    --task classification \
    --model-name GIN \
    --scheduler cosine \
    --epoch 100 \
    --hidden-channel-num 32 \
    --output-channel-num 2 \
    --dropout 0.5 \
    --layer-num 5 \
    --optimizer-name adam \
    --learning-rate 0.00012 \
    --momentum 0.9 \
    --weight-decay 0.0 \
    --scheduler-name CosineAnnealingLR \
    --warmup-epoch 10 \
    --step-size 10 \
    --learning-rate-decay 0.1 \
    --multiplier 1.0 \
    --no-label-data-portion 0.9 \
    --use-pretrained DEFAULT \
    \
    --losses CrossEntropyLoss \
    --channel-weights 1.0 \
    --loss-weights 1.0 \
    \
    --writer-name MUTAG \
    --print-frequency 1 \
    \
    --save-frequency 50 \
    --save-best \
    --init-point 0.1