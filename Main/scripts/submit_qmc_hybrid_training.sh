#!/usr/bin/env bash
#SBATCH -t 2-00:00:00
#SBATCH --gpus-per-node=p100  
#SBATCH --mem=20000
#SBATCH --account=def-rgmelko
#SBATCH --output=outputs/output-%j.out
#SBATCH --mail-user=msmoss@uwaterloo.ca
#SBATCH --mail-type=FAIL,COMPLETED

module load cuda/11.2.2 cudnn/8.2.0

nvidia-smi

export TF_GPU_ALLOCATOR=cuda_malloc_async

source DataEnhancedVMC/bin/activate

echo $delta
echo $data_epochs
echo $vmc_epochs
echo $dim
echo $nh
echo $dset_size

python script_hybrid_training.py \
    $delta \
    $data_epochs \
    $vmc_epochs \
    --vmc_lr $vmc_lr \
    --rnn_dim $dim \
    --nh $nh \
    --seed $seed \
    --qmc_data \
    --dset_size $dset_size