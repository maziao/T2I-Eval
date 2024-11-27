HOST_ADDR=$1
PORT=$2
PWD=`pwd`

python t2i_eval.py \
    --image-root $PWD/data/test \
    --service-url http://${HOST_ADDR}:${PORT}/v1 \
    --output-dir output/minicpm-v-2_6
