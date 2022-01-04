export CUDA_VISIBLE_DEVICES=4

python3 train.py --config configs/vnet/vnet.yml --do_eval  --use_vdl --save_interval 10  --save_dir output --num_workers 2
# python3 -m paddle.distributed.launch train.py --config configs/vnet/vnet.yml --do_eval  --use_vdl --save_interval 500 --save_dir output --num_workers 2
