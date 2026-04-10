#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate iclr

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

echo "=== Step 1: Build distance feature cache (skips already-cached files) ==="
python3 -c "
import glob, os, sys, multiprocessing as mp
import numpy as np
sys.path.insert(0, '.')
from models.transolver_residual.features import precompute_distance_features

all_npz = sorted(glob.glob('gram_data/*.npz'))
real_files = [f for f in all_npz if '.distcache.' not in f]
files = [f for f in real_files if not os.path.exists(f.replace('.npz', '.distcache.npz'))]
print(f'  {len(files)} files need caching (out of {len(real_files)} total)')

def cache_one(path):
    data = np.load(path)
    ia, dist, xsign = precompute_distance_features(
        data['pos'], data['idcs_airfoil'].astype('int64'))
    out = path.replace('.npz', '.distcache.npz')
    np.savez_compressed(out, ia=ia, dist=dist, xsign=xsign)
    return path

if files:
    with mp.Pool(min(32, mp.cpu_count())) as pool:
        for i, path in enumerate(pool.imap_unordered(cache_one, files), 1):
            print(f'  cached {i}/{len(files)}: {os.path.basename(path)}', flush=True)
    print('  Cache complete.')
else:
    print('  All files already cached.')
"

echo ""
echo "=== Step 2: Launch TensorBoard ==="
tensorboard --logdir=runs --port=6008 &> /dev/null &
echo "  TensorBoard running on port 6008"

echo ""
echo "=== Step 3: Train ==="
echo "Starting at $(date)"
python3 train.py \
    --n_layers 4 \
    --hidden_dim 320 \
    --epochs 100 \
    --lr 1e-3 \
    --accum_steps 4 \
    --num_workers 16 \
    --train_fraction 0.9 \
    --augment \
    --run_name run_03
