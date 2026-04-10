#!/bin/bash

# 1. Initialize and activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate iclr  # Be sure to change this if training.py uses a different environment

# 2. Force CUDA to use the PCI Bus ID so it matches nvidia-smi exactly
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 3. Target GPU 4 (The empty NVIDIA L40S)
export CUDA_VISIBLE_DEVICES=4

# Optional: Memory allocator limits (uncomment if your new script is JAX/TensorFlow and needs them)
# export TF_GPU_ALLOCATOR=cuda_malloc_async
# export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Starting training at $(date)"

# 4. Start TensorBoard on a NEW port (6007) so it doesn't collide with your previous run
tensorboard --logdir=runs --port=6007 &

# 5. Run the new script (Replace with the actual path to training.py)
python train.py \
      --n_layers 8 \
      --hidden_dim 256 \
      --epochs 100 \
      --lr 1e-3 \
      --accum_steps 4 \
      --num_workers 32