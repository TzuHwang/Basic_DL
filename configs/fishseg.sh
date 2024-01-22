set -e
set -x

data_root="D:/data/fish/Fish_Dataset/Fish_Dataset"
output_root="./outputs"
config_name="fishseg"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --data-root ${data_root} \
    --output-root ${output_root} \
    --config-name ${config_name} \
    \
    --section training \
    \
    --dataset FishSeg \
    --data-format img \
    --aug sham \
    --crop 0.8 \
    --loading-method all \
    --normalize \
    --maxv 255.0 \
    --batch-size 16 \
    --num-workers 4 \
    --no-label-data-portion 0.8 \
    \
    --task segmentation \
    --model-name UNet \
    --scheduler cosine \
    --epoch 400 \
    --output-channel-num 10 \
    --optimizer-name adam \
    --learning-rate 0.001 \
    --momentum 0.9 \
    --weight-decay 0.0 \
    --scheduler-name CosineAnnealingLR \
    --warmup-epoch 50 \
    --step-size 10 \
    --learning-rate-decay 0.1 \
    --multiplier 1.0 \
    --no-label-data-portion 0.9 \
    \
    --losses SoftDiceLoss CrossentropyND \
    --channel-weights 1.0 \
    --loss-weights 1.0 1.0 \
    \
    --writer-name FishSeg \
    --similarity-fucn SoftDice \
    --print-frequency 10 \
    --rm-exist-log \
    \
    --save-frequency 50 \
    --save-best \
    --init-point 0.1