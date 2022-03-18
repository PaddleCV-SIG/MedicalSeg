# set your GPU ID here
export CUDA_VISIBLE_DEVICES=6

# set the config file name and save directory here
config_name=vnet_mri_spine_seg_128_128_24_15k_12
yml=mri_spine_seg/${config_name}
save_dir_all=saved_model
save_dir=saved_model/${config_name}_0316_1e-2_big_rmresizecrop_class20_12
mkdir -p $save_dir

# Train the model: see the train.py for detailed explanation on script args
python3 train.py --config configs/${yml}.yml \
--save_dir  $save_dir \
--save_interval 500 --log_iters 100 \
--num_workers 6 --do_eval --use_vdl \
--keep_checkpoint_max 5  --seed 0  >> $save_dir/train.log

# # Validate the model: see the val.py for detailed explanation on script args
python3 val.py --config configs/${yml}.yml \
--save_dir  $save_dir/best_model --model_path $save_dir/best_model/model.pdparams

# export the model
python export.py --config configs/${yml}.yml --model_path $save_dir/best_model/model.pdparams

# # infer the model
python deploy/python/infer.py  --config output/deploy.yaml --image_path data/MRSpineSeg/MRI_spine_seg_phase0_class3/images/Case14.npy  --benchmark True
