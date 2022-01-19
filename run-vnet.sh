# set your GPU ID here
export CUDA_VISIBLE_DEVICES=4,5,6

# set the config file name and save directory here
yml=vnet_lung_coronavirus_128_128_128_10k
save_dir=saved_model/${yml}_0112_normalize

# Train the model: see the train.py for detailed explanation on script args
python3 -m paddle.distributed.launch train.py --config configs/lung_coronavirus/${yml}.yml \
--save_dir  $save_dir \
--save_interval 500 --log_iters 100 \
--num_workers 6 --do_eval --use_vdl \
--keep_checkpoint_max 10  --seed 0

# Validate the model: see the val.py for detailed explanation on script args
python3 -m paddle.distributed.launch val.py --config configs/lung_coronavirus/${yml}.yml \
--save_dir  $save_dir/best_model --model_path $save_dir/best_model/model.pdparams
