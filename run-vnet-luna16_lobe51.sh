# set your GPU ID here
export CUDA_VISIBLE_DEVICES=2,3

# set the config file name and save directory here
yml=vnet_luna16_lobe51_128_128_128_15k
save_dir=saved_model/${yml}_0120

# Train the model: see the train.py for detailed explanation on script args
python3 -m paddle.distributed.launch train.py --config configs/luna16_lobe51/${yml}.yml \
--save_dir  $save_dir \
--save_interval 500 --log_iters 100 \
--num_workers 6 --do_eval --use_vdl \
--keep_checkpoint_max 10  --seed 0

# Validate the model: see the val.py for detailed explanation on script args
python3 -m paddle.distributed.launch val.py --config configs/luna16_lobe51/${yml}.yml \
--save_dir  $save_dir/best_model --model_path $save_dir/best_model/model.pdparams
