# #!/bin/bash

# FSS Dataset

joblist=$(sq -h --format="%j")

for delta in 13.455 #$(seq -1.545 0.5 13.455) 
do

    for seed in $(seq 111 111 111) # one seed for now
    do
        X="Hybrid|d=$delta|1D|Nh=32|$seed"
        sbatch -J "$X" --export="delta=$delta,data_epochs=10000,vmc_epochs=10000,t_trans=1000,vmc_lr=5e-5,dim=OneD,nh=32,seed=$seed" submit_hybrid_training.sh

        X="Hybrid|d=$delta|2D|Nh=16|$seed"
        sbatch -J "$X" --export="delta=$delta,data_epochs=3000,vmc_epochs=10000,t_trans=100,vmc_lr=1e-3,dim=TwoD,nh=16,seed=$seed" submit_hybrid_training.sh
            
    done 
    sleep 0.5s

done
