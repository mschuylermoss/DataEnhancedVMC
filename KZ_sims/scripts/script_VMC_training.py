import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import sys
sys.path.append('..')
from dset_helpers import data_given_param
from train import train_wavefunction

#--
import argparse
# Instantiate the parser
parser = argparse.ArgumentParser(description='Parser for starting multiple runs on Graham')
# Required positional argument
parser.add_argument('delta', type=float,
                    help='A required float argument: value of delta') ## dont need -- in front of it
# Required positional argument
parser.add_argument('vmc_epochs', type=int,
                    help='A required integer argument: number of vmc training steps')
# Optional argument
parser.add_argument('--data_lr', type=float, default=1e-3,
                    help='An optional float argument: learning rate for data-driven training')
# Optional argument
parser.add_argument('--vmc_lr', type=float, default=1e-3,
                    help='An optional float argument: learning rate for Hamiltonian-driven training')
# Optional argument
parser.add_argument('--rnn_dim', type=str, default='OneD',
                    help='An optional string argument: dimension of rnn used')
# Optional argument
parser.add_argument('--nh', type=int, default=32,
                    help='An optional integer argument: number of hidden units')
# Optional argument
parser.add_argument('--seed', type=int, default=100,
                    help='An optional integer argument: seed for RNG')
args = parser.parse_args()
#--

Lx = 16
Ly = 16
Omega = 4.24
Rb = 1.15
V0 = Rb**6 * Omega
sweep_rate = 15

delta_arg = args.delta
vmc_steps_arg = args.vmc_epochs
vmc_lr_arg = args.vmc_lr
rnn_dim_arg = args.rnn_dim
nh_arg = args.nh
seed_arg = args.seed

def main():
    config = {
        'name': 'New_Results', 

        'Lx':Lx,  # number of sites in x-direction                    
        'Ly':Ly,  # number of sites in the y-directioni
        'V': V0,
        'Omega': Omega,
        'delta': delta_arg, # detuning
        'sweep_rate':sweep_rate,
        'trunc': 100,
        
        'RNN': rnn_dim_arg, # 1D or 2D RNN wavefunction
        'nh': nh_arg,  # number of memory/hidden units
        'weight_sharing': True,
        'seed': seed_arg, # seed for RNG
        
        'ns': 100,
        'data_epochs':0,
        'vmc_epochs':vmc_steps_arg, # number of Hamiltonian-driven train steps
        'vmc_lr': vmc_lr_arg, # learning rate (for Hamiltonian-driven)
        'ckpt_every': 10,
        
        'Print':True,
        'Write_Data': True,
        'CKPT':True
        }
    
    return train_wavefunction(config) 


if __name__ == "__main__":
    model,e,v,c = main()