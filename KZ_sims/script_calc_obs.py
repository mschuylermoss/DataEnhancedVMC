import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
import sys
sys.path.append('../Main/')
from load_model import LoadModel
from stag_mag import calculate_stag_mag
from sigma_x import calculate_sigma_x
from energy_func import buildlattice, construct_mats
deltas = np.arange(13.455,13.456,0.5)

def calculate_observables(loadmodel_fxn,path_to_config,path_to_ckpts, nsamples, batch_size, stagmag_fxn, O_mat, sigx_fxn):
    
    # Load Model
    model,_,_,_ = loadmodel_fxn(path_to_config,path_to_ckpts)
    
    # Sample & calculate stag mags, sig xs
    if nsamples >= batch_size:
        batches = int((nsamples-(nsamples%batch_size))/batch_size)
        all_stagmags = []
        all_sigma_xs = []
        sigma_xs_per_site = np.zeros((1,256))
        for i in range(batches):
            print(f"Batch #{i}")
            samples,_ = model.sample(batch_size)
            stagmags,_,_,_ = stagmag_fxn(16,16,samples.numpy())
            all_stagmags.append(abs(stagmags))
            sigma_xs,_ = sigx_fxn(O_mat, samples, model.logpsi)
            all_sigma_xs.append(sigma_xs)

    # Take averages
    stagmag = np.mean(all_stagmags)
    sm_var = np.var(all_stagmags)
    sigma_x = np.mean(all_sigma_xs)
    sx_var = np.var(all_sigma_xs)
    
    return stagmag, sm_var, sigma_x, sx_var

sms_vmc1=np.load('./data/observables/sms_vmc1.npy')
sms_vmc1_vars=np.load('./data/observables/sms_vmc1_vars.npy')
sms_h1=np.load('./data/observables/sms_h1.npy')
sms_h1_vars=np.load('./data/observables/sms_h1_vars.npy')
sxs_vmc1=np.load('./data/observables/sxs_vmc1.npy')
sxs_vmc1_vars=np.load('./data/observables/sxs_vmc1_vars.npy')
sxs_h1=np.load('./data/observables/sxs_h1.npy')
sxs_h1_vars=np.load('./data/observables/sxs_h1_vars.npy')

sms_vmc2=np.load('./data/observables/sms_vmc2.npy')
sms_vmc2_vars=np.load('./data/observables/sms_vmc2_vars.npy')
sms_h2=np.load('./data/observables/sms_h2.npy')
sms_h2_vars=np.load('./data/observables/sms_h2_vars.npy')
sxs_vmc2=np.load('./data/observables/sxs_vmc2.npy')
sxs_vmc2_vars=np.load('./data/observables/sxs_vmc2_vars.npy')
sxs_h2=np.load('./data/observables/sxs_h2.npy')
sxs_h2_vars=np.load('./data/observables/sxs_h2_vars.npy')

sms_vmc1=np.append(sms_vmc1,np.load('./data/observables/sms_vmc12.npy'))
sms_vmc1_vars=np.append(sms_vmc1_vars,np.load('./data/observables/sms_vmc1_vars2.npy'))
sms_h1=np.append(sms_h1,np.load('./data/observables/sms_h12.npy'))
sms_h1_vars=np.append(sms_h1_vars,np.load('./data/observables/sms_h1_vars2.npy'))
sxs_vmc1=np.append(sxs_vmc1,np.load('./data/observables/sxs_vmc12.npy'))
sxs_vmc1_vars=np.append(sxs_vmc1_vars,np.load('./data/observables/sxs_vmc1_vars2.npy'))
sxs_h1=np.append(sxs_h1,np.load('./data/observables/sxs_h12.npy'))
sxs_h1_vars=np.append(sxs_h1_vars,np.load('./data/observables/sxs_h1_vars2.npy'))

sms_vmc2=np.append(sms_vmc2,np.load('./data/observables/sms_vmc22.npy'))
sms_vmc2_vars=np.append(sms_vmc2_vars,np.load('./data/observables/sms_vmc2_vars2.npy'))
sms_h2=np.append(sms_h2,np.load('./data/observables/sms_h22.npy'))
sms_h2_vars=np.append(sms_h2_vars,np.load('./data/observables/sms_h2_vars2.npy'))
sxs_vmc2=np.append(sxs_vmc2,np.load('./data/observables/sxs_vmc22.npy'))
sxs_vmc2_vars=np.append(sxs_vmc2_vars,np.load('./data/observables/sxs_vmc2_vars2.npy'))
sxs_h2=np.append(sxs_h2,np.load('./data/observables/sxs_h22.npy'))
sxs_h2_vars=np.append(sxs_h2_vars,np.load('./data/observables/sxs_h2_vars2.npy'))

scratch_path = '/scratch/msmoss/RNN_sims/Rydbergs/N_256/New_Results/'
train_quantities_path = scratch_path + 'train_quantities/'
interactions = buildlattice(16,16,trunc=100)
omega_matrix,_,_ = construct_mats(interactions,256)

for delta in deltas:
    delta = round(delta,3)
    print(delta)
    path_vmc1 = train_quantities_path + f'OneD_rnn/delta_{delta}/seed_111/vmc_only/'
    path_vmc1_ckpts = scratch_path + f'OneD_rnn/delta_{delta}/seed_111/vmc_only/'
    path_h1 = train_quantities_path + f'OneD_rnn/delta_{delta}/seed_111/hybrid_train/1000_ds/lr_5e-05/'
    path_h1_ckpts = scratch_path + f'OneD_rnn/delta_{delta}/seed_111/hybrid_train/1000_ds/lr_5e-05/'
    
    path_vmc2 = train_quantities_path + f'TwoD_rnn/delta_{delta}/seed_111/vmc_only/'
    path_vmc2_ckpts = scratch_path + f'TwoD_rnn/delta_{delta}/seed_111/vmc_only/'
    path_h2 = train_quantities_path + f'TwoD_rnn/delta_{delta}/seed_111/hybrid_train/100_ds/lr_0.001/'
    path_h2_ckpts = scratch_path + f'TwoD_rnn/delta_{delta}/seed_111/hybrid_train/100_ds/lr_0.001/'
    
    sm_vmc1, sm_var_vmc1, sx_vmc1, sx_var_vmc1 = calculate_observables(LoadModel,path_vmc1,path_vmc1_ckpts, 1000, 100, calculate_stag_mag, omega_matrix, calculate_sigma_x)
    sm_h1, sm_var_h1, sx_h1, sx_var_h1 = calculate_observables(LoadModel,path_h1,path_h1_ckpts, 1000, 100, calculate_stag_mag, omega_matrix, calculate_sigma_x)
    sm_vmc2, sm_var_vmc2, sx_vmc2, sx_var_vmc2 = calculate_observables(LoadModel,path_vmc2,path_vmc2_ckpts, 1000, 100, calculate_stag_mag, omega_matrix, calculate_sigma_x)
    sm_h2, sm_var_h2, sx_h2, sx_var_h2 = calculate_observables(LoadModel,path_h2,path_h2_ckpts, 1000, 100, calculate_stag_mag, omega_matrix, calculate_sigma_x)

    sms_vmc1 = np.append(sms_vmc1,sm_vmc1)
    sms_vmc1_vars = np.append(sms_vmc1_vars,sm_var_vmc1)
    sms_h1 = np.append(sms_h1,sm_h1)
    sms_h1_vars = np.append(sms_h1_vars,sm_var_h1)
    sxs_vmc1 = np.append(sxs_vmc1,sx_vmc1)
    sxs_vmc1_vars = np.append(sxs_vmc1_vars,sx_var_vmc1)
    sxs_h1 = np.append(sxs_h1,sx_h1)
    sxs_h1_vars = np.append(sxs_h1_vars,sx_var_h1)
    
    sms_vmc2 = np.append(sms_vmc2,sm_vmc2)
    sms_vmc2_vars = np.append(sms_vmc2_vars,sm_var_vmc2)
    sms_h2 = np.append(sms_h2,sm_h2)
    sms_h2_vars = np.append(sms_h2_vars,sm_var_h2)
    sxs_vmc2 = np.append(sxs_vmc2,sx_vmc2)
    sxs_vmc2_vars = np.append(sxs_vmc2_vars,sx_var_vmc2)
    sxs_h2 = np.append(sxs_h2,sx_h2)
    sxs_h2_vars = np.append(sxs_h2_vars,sx_var_h2)

    np.save('./data/observables/sms_vmc1_all',sms_vmc1)
    np.save('./data/observables/sms_vmc1_vars_all',sms_vmc1_vars)
    np.save('./data/observables/sms_h1_all',sms_h1)
    np.save('./data/observables/sms_h1_vars_all',sms_h1_vars)
    np.save('./data/observables/sxs_vmc1_all',sxs_vmc1)
    np.save('./data/observables/sxs_vmc1_vars_all',sxs_vmc1_vars)
    np.save('./data/observables/sxs_h1_all',sxs_h1)
    np.save('./data/observables/sxs_h1_vars_all',sxs_h1_vars)

    np.save('./data/observables/sms_vmc2_all',sms_vmc2)
    np.save('./data/observables/sms_vmc2_vars_all',sms_vmc2_vars)
    np.save('./data/observables/sms_h2_all',sms_h2)
    np.save('./data/observables/sms_h2_vars_all',sms_h2_vars)
    np.save('./data/observables/sxs_vmc2_all',sxs_vmc2)
    np.save('./data/observables/sxs_vmc2_vars_all',sxs_vmc2_vars)
    np.save('./data/observables/sxs_h2_all',sxs_h2)
    np.save('./data/observables/sxs_h2_vars_all',sxs_h2_vars)

    print(f"Saved observables for delta = {delta}")