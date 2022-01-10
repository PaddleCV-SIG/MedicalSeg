export CUDA_VISIBLE_DEVICES=6,7

yml=unet3d_lung_coronavirus_128_128_128_10k
save_dir=saved_model/${yml}_0110_addce

python3 -m paddle.distributed.launch train.py --config configs/unet3d/${yml}.yml \
--save_dir  $save_dir \
--save_interval 500 --log_iters 100 \
--num_workers 2 --do_eval \
--keep_checkpoint_max 10  --seed 0 \

# python3 -m paddle.distributed.launch train.py --config configs/vnet/vnet.yml --do_eval  --use_vdl --save_interval 500 --save_dir output --num_workers 2
