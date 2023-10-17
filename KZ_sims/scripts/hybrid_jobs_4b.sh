# #!/bin/bash

# FSS Dataset

joblist=$(sq -h --format="%j")

for delta in -1.545 4.455 4.955 13.455
do
    for seed in $(seq 111 111 111) # one seed for now
    do
        for t_trans in $(seq 20 20 80)
        do
            X="Hybrid4b|d=$delta|2D|Nh=16|$seed"
            sbatch -J "$X" --export="delta=$delta,data_epochs=250,vmc_epochs=3000,t_trans=$t_trans,vmc_lr=1e-3,dim=TwoD,nh=16,seed=$seed" submit_hybrid_training.sh
        done

        for t_trans in $(seq 120 20 200)
        do
            X="Hybrid4b|d=$delta|2D|Nh=16|$seed"
            sbatch -J "$X" --export="delta=$delta,data_epochs=250,vmc_epochs=3000,t_trans=$t_trans,vmc_lr=1e-3,dim=TwoD,nh=16,seed=$seed" submit_hybrid_training.sh
        done
    done 
    sleep 0.5s

done
