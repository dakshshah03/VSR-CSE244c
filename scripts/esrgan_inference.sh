MODEL_INFERENCE_SCRIPT="./../models/esrgan_test.py"
DATA_PATH="./../data/REDS/benchmark_data/low_resolution"
MODEL="esrgan"

python $MODEL_INFERENCE_SCRIPT --input_dir $DATA_PATH/000/ --output_dir ./../output/$MODEL/000/
python $MODEL_INFERENCE_SCRIPT --input_dir $DATA_PATH/011/ --output_dir ./../output/$MODEL/011/
python $MODEL_INFERENCE_SCRIPT --input_dir $DATA_PATH/015/ --output_dir ./../output/$MODEL/015/
python $MODEL_INFERENCE_SCRIPT --input_dir $DATA_PATH/020/ --output_dir ./../output/$MODEL/020/

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
    --fps 30

python frame_merge.py \
    -i ./../output/$MODEL/020/ \
    -o ./../output/$MODEL/020_merged.mp4 \
    --fps 30