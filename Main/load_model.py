import os
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
from dset_helpers import load_exact_Es
from OneD_RNN import OneD_RNN_wavefxn, RNNWavefunction1D
from TwoD_RNN import MDRNNWavefunction,MDTensorizedRNNCell,MDRNNGRUcell
from energy_func import buildlattice,construct_mats,get_Rydberg_Energy_Vectorized
from stag_mag import calculate_stag_mag

def optimizer_initializer(optimizer):
    fake_var = tf.Variable(1.0)
    with tf.GradientTape() as tape:
        fake_loss = tf.reduce_sum(fake_var ** 2)
    grads = tape.gradient(fake_loss, [fake_var])
    optimizer.apply_gradients(zip(grads, [fake_var]))

def LoadModel(path_to_config,path_to_ckpts):

    '''
    Load a configuration from path_to_config. 
    Initialize the RNN wavefunction using the config file.
    Reload the model from the checkpoints in path_to_ckpts.
    Return the mode and the saved training quantities.
    '''
    # ---- Load Config -----------------------------------------------------------------------
    config = {}
    with open(path_to_config+"config.txt") as f:
        for line in f:
            (key, val) = line.strip().split('=')
            if key=='name' or key=='RNN':
                val = str(val)
            if key=='Lx' or key=='Ly' or key=='sweep_rate' or key=='nh' or key=='trunc' or key=='seed' or key=='VMC_epochs' or key=='Data_epochs' or key=='ns':
                val = int(val)
            if key=='V'or key=='Omega' or key=='delta' or key=='vmc_lr':
                val = float(val)
            if key=='weight_sharing' or key=='MDGRU' or key=='batch_samples' or key=='Print' or key=='Write_Data' or key=='CKPT':
                val = bool(key)
            config[key] = val
            config['Print'] = True

    # ---- System Parameters -----------------------------------------------------------------
    Lx = config['Lx']
    Ly = config['Ly']
    
    # ---- RNN Parameters ---------------------------------------------------------------------
    num_hidden = config['nh']
    learning_rate = config['vmc_lr']
    weight_sharing = config['weight_sharing']
    seed = config['seed']
    rnn_type = config['RNN']

    # ---- Training Parameters ----------------------------------------------------------------
    global_step = tf.Variable(0, name="global_step")

    # ---- Initiate RNN Wave Function ----------------------------------------------------------
    if config['RNN'] == 'OneD':
        if config['Print'] ==True:
            print(f"Training a one-D RNN wave function with {num_hidden} hidden units and shared weights.")
        wavefxn = OneD_RNN_wavefxn(Lx,Ly,num_hidden,learning_rate,seed)
    elif config['RNN'] =='TwoD':
        if config['Print'] ==True:
            print(f"Training a two-D RNN wave function with {num_hidden} hidden units and shared weights = {weight_sharing}.")
        wavefxn = MDRNNWavefunction(Lx,Ly,num_hidden,learning_rate,weight_sharing,seed,cell=MDRNNGRUcell)
    else:
        raise ValueError(f"{config['RNN']} is not a valid option for the RNN wave function. Please choose OneD or TwoD.")

    # ---- Reload From CKPT -------------------------------------------------------------
    ckpt = tf.train.Checkpoint(step=global_step, variables=wavefxn.trainable_variables)
    manager = tf.train.CheckpointManager(ckpt, path_to_ckpts, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Checkpoint found.")
        print(f"Restored from {manager.latest_checkpoint}.")
        latest_ckpt = ckpt.step.numpy()
        optimizer_initializer(wavefxn.optimizer)
        print(f"The final step was {latest_ckpt}.")
        energy = np.load(path_to_config+'/Energy.npy').tolist()[0:latest_ckpt]
        variance = np.load(path_to_config+'/Variance.npy').tolist()[0:latest_ckpt]
        cost = np.load(path_to_config+'/Cost.npy').tolist()[0:latest_ckpt]

    else:
        raise ValueError("No checkpoint found.")
                
    return wavefxn, energy, variance, cost