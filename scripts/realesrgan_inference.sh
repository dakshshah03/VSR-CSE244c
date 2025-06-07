MODEL_INFERENCE_SCRIPT="./../submodules/Real-ESRGAN/inference_realesrgan.py"
DATA_PATH="./../data/REDS/benchmark_data/low_resolution"
MODEL="realesr"

python $MODEL_INFERENCE_SCRIPT -n RealESRGAN_x4plus -i $DATA_PATH/000/ -o ./../output/$MODEL/000
python $MODEL_INFERENCE_SCRIPT -n RealESRGAN_x4plus -i $DATA_PATH/011/ -o ./../output/$MODEL/011
python $MODEL_INFERENCE_SCRIPT -n RealESRGAN_x4plus -i $DATA_PATH/015/ -o ./../output/$MODEL/015
python $MODEL_INFERENCE_SCRIPT -n RealESRGAN_x4plus -i $DATA_PATH/020/ -o ./../output/$MODEL/020

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