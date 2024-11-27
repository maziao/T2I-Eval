CUDA=$1
NUM_GPU=$2

NPROC_PER_NODE=${NUM_GPU} \
CUDA_VISIBLE_DEVICES=${CUDA} swift sft \
    --model_type minicpm-v-v2_6-chat \
    --model_id_or_path models/minicpm-v-2_6/original \
    --deepspeed default-zero2 \
    --max_length 4096 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --use_flash_attn true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --dataset_test_ratio 0.0 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --dataset data/train/minicpm-v-2_6/train.json \
    --output_dir models/minicpm-v-2_6/checkpoint \
