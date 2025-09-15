
cd /root/FreeNeRF

method=risenerf

num_shots=3
project=llff${num_shots}-$method
scans=(flower fortress horns leaves orchids room trex)
for scan in "${scans[@]}" # 配列の全要素をループ
do
export CUDA_VISIBLE_DEVICES=0,1,2,3
    python3 train.py \
        --gin_configs configs/$method/llff${num_shots}_$method.gin \
        --gin_bindings "Config.llff_scan = '$scan'" \
        --gin_bindings "Config.expname = '$scan-train'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.project = '$project'" \
        --gin_bindings "Config.render_chunk_size = 16384" 

    python3 eval.py \
        --gin_configs configs/$method/llff${num_shots}_$method.gin \
        --gin_bindings "Config.llff_scan = '$scan'" \
        --gin_bindings "Config.expname = '$scan-eval'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.log_img_to_wandb = True" \
        --gin_bindings "Config.project = '$project'"
done

num_shots=6
project=llff${num_shots}-$method
scans=(fern flower fortress horns leaves orchids room trex)
for scan in "${scans[@]}" # 配列の全要素をループ
do
export CUDA_VISIBLE_DEVICES=0,1,2,3
    python3 train.py \
        --gin_configs configs/$method/llff${num_shots}_$method.gin \
        --gin_bindings "Config.llff_scan = '$scan'" \
        --gin_bindings "Config.expname = '$scan-train'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.project = '$project'" \
        --gin_bindings "Config.render_chunk_size = 16384" 

    python3 eval.py \
        --gin_configs configs/$method/llff${num_shots}_$method.gin \
        --gin_bindings "Config.llff_scan = '$scan'" \
        --gin_bindings "Config.expname = '$scan-eval'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.log_img_to_wandb = True" \
        --gin_bindings "Config.project = '$project'"
done

num_shots=9
project=llff${num_shots}-$method
scans=(fern flower fortress horns leaves orchids room trex)
for scan in "${scans[@]}" # 配列の全要素をループ
do
export CUDA_VISIBLE_DEVICES=0,1,2,3
    python3 train.py \
        --gin_configs configs/$method/llff${num_shots}_$method.gin \
        --gin_bindings "Config.llff_scan = '$scan'" \
        --gin_bindings "Config.expname = '$scan-train'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.project = '$project'" \
        --gin_bindings "Config.render_chunk_size = 16384" 

    python3 eval.py \
        --gin_configs configs/$method/llff${num_shots}_$method.gin \
        --gin_bindings "Config.llff_scan = '$scan'" \
        --gin_bindings "Config.expname = '$scan-eval'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.log_img_to_wandb = True" \
        --gin_bindings "Config.project = '$project'"
done