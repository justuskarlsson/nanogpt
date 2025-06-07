


num_nodes=${1:-4}
OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=$num_nodes pretraining.py
