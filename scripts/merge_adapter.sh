CUDA=$1

CUDA_VISIBLE_DEVICES=${CUDA} swift export \
    --model_type minicpm-v-v2_6-chat \
    --model_id_or_path models/minicpm-v-2_6/original \
    --ckpt_dir models/minicpm-v-2_6/LoRA \
    --merge_lora true
