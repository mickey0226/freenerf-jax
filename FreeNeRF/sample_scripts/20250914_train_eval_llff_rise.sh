
cd /root/FreeNeRF

method=risenerf
num_shots=3
project=llff${num_shots}-$method-degview5-layer2
scans=(room fortress)
for scan in "${scans[@]}" # 配列の全要素をループ
do
export CUDA_VISIBLE_DEVICES=0,1,2,3
    python3 train.py \
        --gin_configs configs/$method/llff${num_shots}_$method.gin \
        --gin_bindings "Config.llff_scan = '$scan'" \
        --gin_bindings "Config.expname = '$scan-train'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.project = '$project'" \
        --gin_bindings "MLP.net_depth_viewdirs = 2" \
        --gin_bindings "Config.grad_max_norm = 1.0" \
        --gin_bindings "Config.render_chunk_size = 16384" 
    python3 eval.py \
        --gin_configs configs/$method/llff${num_shots}_$method.gin \
        --gin_bindings "Config.llff_scan = '$scan'" \
        --gin_bindings "Config.expname = '$scan-eval'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.log_img_to_wandb = True" \
        --gin_bindings "MLP.net_depth_viewdirs = 2" \
        --gin_bindings "Config.grad_max_norm = 1.0" \
        --gin_bindings "Config.project = '$project'"
done

method=risenerf
num_shots=3
project=llff${num_shots}-$method-degview5-layer3
scans=(room fortress)
for scan in "${scans[@]}" # 配列の全要素をループ
do
export CUDA_VISIBLE_DEVICES=0,1,2,3
    python3 train.py \
        --gin_configs configs/$method/llff${num_shots}_$method.gin \
        --gin_bindings "Config.llff_scan = '$scan'" \
        --gin_bindings "Config.expname = '$scan-train'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.project = '$project'" \
        --gin_bindings "MLP.net_depth_viewdirs = 3" \
        --gin_bindings "Config.grad_max_norm = 1.0" \
        --gin_bindings "Config.render_chunk_size = 16384" 
    python3 eval.py \
        --gin_configs configs/$method/llff${num_shots}_$method.gin \
        --gin_bindings "Config.llff_scan = '$scan'" \
        --gin_bindings "Config.expname = '$scan-eval'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.log_img_to_wandb = True" \
        --gin_bindings "MLP.net_depth_viewdirs = 3" \
        --gin_bindings "Config.grad_max_norm = 1.0" \
        --gin_bindings "Config.project = '$project'"
done

method=risenerf
num_shots=3
project=llff${num_shots}-$method-degview5-layer4
scans=(room fortress)
for scan in "${scans[@]}" # 配列の全要素をループ
do
export CUDA_VISIBLE_DEVICES=0,1,2,3
    python3 train.py \
        --gin_configs configs/$method/llff${num_shots}_$method.gin \
        --gin_bindings "Config.llff_scan = '$scan'" \
        --gin_bindings "Config.expname = '$scan-train'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.project = '$project'" \
        --gin_bindings "MLP.net_depth_viewdirs = 4" \
        --gin_bindings "Config.grad_max_norm = 1.0" \
        --gin_bindings "Config.render_chunk_size = 16384" 
    python3 eval.py \
        --gin_configs configs/$method/llff${num_shots}_$method.gin \
        --gin_bindings "Config.llff_scan = '$scan'" \
        --gin_bindings "Config.expname = '$scan-eval'" \
        --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
        --gin_bindings "Config.log_img_to_wandb = True" \
        --gin_bindings "MLP.net_depth_viewdirs = 4" \
        --gin_bindings "Config.grad_max_norm = 1.0" \
        --gin_bindings "Config.project = '$project'"
done