export CUDA_VISIBLE_DEVICES=4,5,6

yml=vnet_lung_coronavirus_128_128_128_10k
save_dir=saved_model/${yml}_0111_inc_iters_syncbn

python3 -m paddle.distributed.launch train.py --config configs/lung_coronavirus/${yml}.yml \
--save_dir  $save_dir \
--save_interval 500 --log_iters 100 \
--num_workers 6 --do_eval --use_vdl\
--keep_checkpoint_max 10  --seed 0 \
