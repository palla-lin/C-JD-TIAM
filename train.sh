#CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 
python code/train.py\
 -bs 1024\
 -ds 0\
 -run_name vx\
 -output_dir output\
 -num_layers 12 2>&1 | tee ./consolelog1.txt