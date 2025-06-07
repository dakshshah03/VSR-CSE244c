MODEL_INFERENCE_SCRIPT="./baseline.py"
DATA_PATH="./../data/REDS/benchmark_data/low_resolution"
MODEL="baseline"

python $MODEL_INFERENCE_SCRIPT -i $DATA_PATH/000/ -o ./../output/$MODEL/000 -s 4
python $MODEL_INFERENCE_SCRIPT -i $DATA_PATH/011/ -o ./../output/$MODEL/011 -s 4
python $MODEL_INFERENCE_SCRIPT -i $DATA_PATH/015/ -o ./../output/$MODEL/015 -s 4
python $MODEL_INFERENCE_SCRIPT --i $DATA_PATH/020/ -o ./../output/$MODEL/020 -s 4

python frame_merge.py \
    -i ./../output/$MODEL/000/ \
    -o ./../output/$MODEL/000_merged.mp4 \
    --fps 30

python frame_merge.py \
    -i ./../output/$MODEL/011/ \
    -o ./../output/$MODEL/011_merged.mp4 \
    --fps 30

python frame_merge.py \
    -i ./../output/$MODEL/015/ \
    -o ./../output/$MODEL/015_merged.mp4 \
    --fps 30\

python frame_merge.py \
    -i ./../output/$MODEL/020/ \
    -o ./../output/$MODEL/020_merged.mp4 \
    --fps 30