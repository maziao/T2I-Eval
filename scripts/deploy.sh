CUDA=$1
HOST_ADDR=$2
PORT=$3

CUDA_VISIBLE_DEVICES=${CUDA} vllm serve models/minicpm-v-2_6/LoRA-merged \
  --host ${HOST_ADDR} \
  --port ${PORT} \
  --trust-remote-code \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=2 \
  --enforce-eager
