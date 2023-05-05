import os
import tensorflow as tf
import numpy as np
import glob
# import pandas as pd

Exact_Es = {'4':-0.4534132086591546,'8':-0.40518005298872917,'12':-0.3884864748124427,'16':-0.380514770608724}

def load_exact_Es(dim): ## note that the exact QMC energies are stored in info.txt for each of the dims we looked at in our last project
    path = "../QMC_data"
    dim_path = f"Dim={dim}_M=1000000_V=7_omega=1.0_delta=1.0" # Can change this to look at Dim = 4, 8, 12, 16
    exact_e = Exact_Es[f'{dim}']
    exact_e = Exact_Es[f'{dim}']
    return exact_e

def load_QMC_data(dim):
    path = "../../QMC_data"
    dim_path = f"Dim={dim}_M=1000000_V=7_omega=1.0_delta=1.0" # Can change this to look at Dim = 4, 8, 12, 16
    files_we_want = glob.glob(os.path.join(path,dim_path,'samples*'))
    uploaded = {}
    for file in files_we_want:
        data = np.loadtxt(file)
        uploaded[file] = data
    return uploaded

# def load_KZ_QMC_data(delta):
#     path = "./../../../QMC_data/all_samples/"
#     delta_path = f"delta_{delta}/"
#     data = np.zeros((1,256))
#     for M in np.array([100000,150000,200000,300000]):
#         for seed in np.arange(100,2501,100):
#             for batch in np.arange(1,21,1):
#                 batch_s = '0'+str(batch)
#                 batch_s = batch_s[-2:]
#                 samples = np.array(pd.read_csv(path+delta_path+f'samples_M=({M})_seed=({seed})_batch=({batch_s}).csv', sep=','))
#                 data = np.append(data,samples,axis=0)
#     return data[1:,:]

def load_KZ_QMC_uncorr_data(delta,dset_size):
    data = np.load(f"./../../../QMC_data/all_samples/delta_{delta}/all_samples_delta_{delta}.npy")
    indices = np.random.randint(0,high=np.shape(data)[0],size=dset_size)
    uncorr_data = data[indices,:]
    return uncorr_data

def load_KZ_QMC_uncorr_data_from_batches(delta,dset_size):
    data = np.zeros((1,256))
    for i in range(0,10):
        batch = np.load(f"./../../../QMC_data/all_samples/delta_{delta}/all_samples_batch_{i}.npy")
        data = np.append(data, batch,axis = 0)
    data = data[1:,:]
    indices = np.random.randint(0,high=np.shape(data)[0],size=dset_size)
    uncorr_data = data[indices,:]
    return uncorr_data.astype(int)

def create_KZ_QMC_tf_dataset(data):
    return tf.data.Dataset.from_tensor_slices(data)

def create_tf_dataset_from_QMCdata(uploaded_files, data_step_size=100):
    '''
    create tensor flow data set from uploaded files
    data_step_size (int): determines step size when loading data (for QMC need to have this to overcome autocorrelations)
    '''
    data = []
    for file in uploaded_files:
        new_data = uploaded_files[file]
        new_data = new_data.astype(int)
        new_data = new_data[::data_step_size]
        data.extend(new_data)

    #convert to tf.data.Dataset
    data = np.array(data)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return dataset

def create_KZ_tf_dataset(data):
    # The data comes in shape ((lattice,shape),1,shots) where (lattice,shape) = square lattice (EX: 16x16)
    Lx = np.shape(data)[0]
    shots = np.shape(data)[-1]
    data = np.reshape(data,(Lx**2,shots))
    data = data.T
    tf_data = tf.data.Dataset.from_tensor_slices(data)
    return tf_data

def bool_to_bin(rydberg_dataset):
    return rydberg_dataset.astype(int)

def data_given_param(sweep_rate:int,delta_value,Lx):
    data_files = np.load(f'../../KZ_Data/{Lx}x{Lx}/KZ_data_{Lx}x{Lx}_{sweep_rate}_MHz_per_us.npz')
    param_value = delta_value/data_files['rabi_freq']
    index = np.where(np.isclose(data_files['params'],param_value))[0]
    if len(index) < 1:
        return print("This is not a valid value of delta!")
    data = data_files['rydberg_data'][:,:,index,:]
    data = bool_to_bin(data)
    return data
