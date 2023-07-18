#!/usr/bin/env bash
#SBATCH -t 0-16:00:00
#SBATCH --gpus-per-node=p100  
#SBATCH --mem=20000
#SBATCH --account=def-rgmelko
#SBATCH --output=outputs/output-%j.out
#SBATCH --mail-user=msmoss@uwaterloo.ca
#SBATCH --mail-type=FAIL

module load cuda/11.2.2 cudnn/8.2.0

nvidia-smi

export TF_GPU_ALLOCATOR=cuda_malloc_async

source DataEnhancedVMC/bin/activate

echo $delta
echo $dim
echo $nh

python script_VMC_training.py \
    $delta \
    10000 \
    --vmc_lr 1e-3 \
    --rnn_dim $dim \
    --nh $nh \
    --seed $seed
