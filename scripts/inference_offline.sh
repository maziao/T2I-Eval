CUDA=$1
PWD=`pwd`

CUDA_VISIBLE_DEVICES=${CUDA} python t2i_eval_offline.py \
    --image-root $PWD/data/test \
    --model-name-or-path models/minicpm-v-2_6/LoRA-merged \
    --output-dir output/minicpm-v-2_6-offline
