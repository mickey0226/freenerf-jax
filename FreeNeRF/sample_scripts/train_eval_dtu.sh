cd /root/FreeNeRF
method=freenerf
num_shots=3
# In DTU, following scans are used
# 8, 21, 30, 31, 34, 38, 40, 4145, 55, 63, 82, 103, 110, 114
scan=8
project=dtu${num_shots}-$method

# to overwrite the max_steps in the gin config file, use the following line
#   --gin_bindings "Config.freq_reg_end = $max_steps" 

export CUDA_VISIBLE_DEVICES=0,1,2,3
# python3 train.py \
#     --gin_configs configs/$method/dtu${num_shots}_${method}.gin \
#     --gin_bindings "Config.dtu_scan = 'scan$scan'" \
#     --gin_bindings "Config.expname = '$scan-train'" \
#     --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
#     --gin_bindings "Config.project = '$project'" \
#     --gin_bindings "Config.render_chunk_size = 16384" 

python3 eval.py \
    --gin_configs configs/$method/dtu${num_shots}_${method}.gin \
    --gin_bindings "Config.dtu_scan = 'scan$scan'" \
    --gin_bindings "Config.expname = '$scan-eval'" \
    --gin_bindings "Config.checkpoint_dir = 'out/$method/$project/$scan'" \
    --gin_bindings "Config.log_img_to_wandb = True" \
    --gin_bindings "Config.project = '$project'"