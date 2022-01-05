export CUDA_VISIBLE_DEVICES=1,2

# yml = vnet_lung_coronavirus_128_128_128_40k

python3 -m paddle.distributed.launch train.py --config configs/vnet/vnet_lung_coronavirus_128_128_128_40k.yml --save_dir  saved_model/vnet_lung_coronavirus_128_128_128_40k \
--save_interval 500 --log_iters 100 \
--num_workers 2 --do_eval \
--keep_checkpoint_max 10  --seed 0 \

# python3 -m paddle.distributed.launch train.py --config configs/vnet/vnet.yml --do_eval  --use_vdl --save_interval 500 --save_dir output --num_workers 2
