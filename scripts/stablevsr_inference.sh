MODEL_INFERENCE_SCRIPT="./../submodules/StableVSR/test.py"
DATA_PATH="./../data/REDS/benchmark_data/low_resolution"
MODEL="stablevsr"

python $MODEL_INFERENCE_SCRIPT --in_path $DATA_PATH --out_path ./../output/$MODEL/ --num_inference_steps 50 

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