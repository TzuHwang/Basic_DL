set -e
set -x

data_root="D:/data/Dog_vs_Cat"
output_root="./outputs"
config_name="dogcatcls_v8m_bce"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --data-root ${data_root} \
    --output-root ${output_root} \
    --config-name ${config_name} \
    \
    --section training \
    \
    --dataset DogCatCls \
    --data-format img \
    --aug default \
    --crop 0.8 \
    --image_size 224 \
    --loading-method all \
    --normalize \
    --maxv 255.0 \
    --batch-size 64 \
    --num-workers 4 \
    --no-label-data-portion 0.8 \
    \
    --task classification \
    --model-name YOLO_v8m \
    --scheduler cosine \
    --epoch 400 \
    --output-channel-num 2 \
    --dropout 0.0 \
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
    --use-pretrained DEFAULT \
    \
    --losses BCELoss \
    --channel-weights 1.0 \
    --loss-weights 1.0 \
    \
    --writer-name DogCatCls \
    --print-frequency 60 \
    \
    --save-frequency 50 \
    --save-best \
    --init-point 0.1