cd /root/FreeNeRF
method=risenerf

num_shots=3
scans=(8 21 30 31 34 38 40 41 45 55 63 82 103 110 114)
project=dtu${num_shots}-$method
for scan in "${scans[@]}" # 配列の全要素をループ
do
export CUDA_VISIBLE_DEVICES=0,1,2,3
    python3 train.py \
        --gin_configs configs/$method/dtu${num_shots}_${method}.gin \
        --gin_bindings "Config.dtu_scan = 'scan$scan'" \
        --gin_bindings "Config.expname = '$scan-train'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.project = '$project'" \
        --gin_bindings "Config.render_chunk_size = 16384" 

    python3 eval.py \
        --gin_configs configs/$method/dtu${num_shots}_${method}.gin \
        --gin_bindings "Config.dtu_scan = 'scan$scan'" \
        --gin_bindings "Config.expname = '$scan-eval'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.log_img_to_wandb = True" \
        --gin_bindings "Config.project = '$project'"
done