# #!/bin/bash

# FSS Dataset

joblist=$(sq -h --format="%j")

for delta in $(seq -1.545 0.5 13.455) 
do

    for seed in $(seq 111 111 111) # one seed for now
    do
        for dset_size in 1000 10000
        do
            X="Hybrid|d=$delta|1D|Nh=32|$seed"
            sbatch -J "$X" --export="delta=$delta,data_epochs=10000,vmc_epochs=10000,vmc_lr=1e-4,dim=OneD,nh=32,seed=$seed,dset_size=$dset_size" submit_qmc_hybrid_training.sh

            X="Hybrid|d=$delta|2D|Nh=16|$seed"
            sbatch -J "$X" --export="delta=$delta,data_epochs=5000,vmc_epochs=10000,vmc_lr=1e-3,dim=TwoD,nh=16,seed=$seed,dset_size=$dset_size" submit_qmc_hybrid_training.sh
        done

    done 
    sleep 0.5s

done
