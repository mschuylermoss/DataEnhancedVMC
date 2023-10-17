#!/usr/bin/env bash
#SBATCH -t 0-01:00:00
#SBATCH --mem=20000
#SBATCH --account=def-rgmelko
#SBATCH --output=outputs/output-%j.out
#SBATCH --mail-user=msmoss@uwaterloo.ca
#SBATCH --mail-type=FAIL

source ../Main/scripts/DataEnhancedVMC/bin/activate

python script_calc_obs.py
