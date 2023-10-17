# #!/bin/bash

# FSS Dataset

joblist=$(sq -h --format="%j")

for delta in -1.545 4.455 4.955 13.455
do

    for seed in $(seq 222 111 333) # one seed for now
    do
        for dset_size in 10000
        do
            # X="QMC$dset_size|d=$delta|1D|Nh=32|$seed"
            # sbatch -J "$X" --export="delta=$delta,data_epochs=10000,vmc_epochs=10000,vmc_lr=5e-5,dim=OneD,nh=32,seed=$seed,dset_size=$dset_size" submit_hybrid_training_qmc.sh

            X="QMC$dset_size|d=$delta|2D|Nh=16|$seed"
            sbatch -J "$X" --export="delta=$delta,data_epochs=3000,vmc_epochs=10000,vmc_lr=1e-3,dim=TwoD,nh=16,seed=$seed,dset_size=$dset_size" submit_hybrid_training_qmc.sh
        done

    done 
    sleep 0.5s

done
