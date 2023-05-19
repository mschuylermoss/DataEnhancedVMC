import tensorflow as tf
import numpy as np
from dset_helpers import create_KZ_tf_dataset, data_given_param
from dset_helpers import load_KZ_QMC_uncorr_data_from_batches, create_KZ_QMC_tf_dataset
from OneD_RNN import OneD_RNN_wavefxn 
from TwoD_RNN import MDRNNWavefunction, MDTensorizedRNNCell, MDRNNGRUcell
from energy_func import buildlattice, construct_mats, get_Rydberg_Energy_Vectorized
from helpers import optimizer_initializer, min_moving_average, write_config, load_vmc_start
# from stag_mag import calculate_stag_mag

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def train_wavefunction(config):

    '''
    This function takes a configuration of parameters and trains an RNN wavefunction to learn
    the ground state of a Rydberg atom array. This function will train the RNN with:
        -VMC only           (set data_epochs = 0, vmc_epochs > 0)
        -Data only          (set data_epochs > 0, vmc_epochs = 0)
        -Data-enhanced VMC  (set data_epochs > 0, vmc_epochs > 0)

    Note: For data-enhanced VMC, this code will train the model using this appropriate data creating
    regular benchmarks. Once the data-driven training is complete, the lowest energy reached 
    by the trained model will be found and variational training will begin from *that* point.
    Therefore, it is helpful if data_epochs is an integer between 3,000-10,000.
    '''

    # ---- System Parameters -----------------------------------------------------------------
    Lx = config['Lx']
    Ly = config['Ly']
    V = config['V']
    Omega = config['Omega']
    delta = config['delta']
    sweep_rate = config['sweep_rate']
    trunc = config.get('trunc',100)
    if config['Print'] ==True:
        print(f"The system is an array of {Lx} by {Ly} Rydberg Atoms.")
        print(f"The experimental parameters are: V = {V}, delta = {delta}, Omega = {Omega}.")

    # ---- RNN Parameters ---------------------------------------------------------------------
    rnn_type = config['RNN']
    num_hidden = config['nh']
    weight_sharing = config['weight_sharing']
    seed = config['seed']
    if config['Print'] ==True:
        print(f"Training a {rnn_type} RNN wavefunction with {num_hidden} hidden units.")

    # ---- Initiate RNN Wave Function ----------------------------------------------------------
    if rnn_type == 'OneD':
        wavefxn = OneD_RNN_wavefxn(Lx,Ly,num_hidden,1e-3,seed)
    elif rnn_type =='TwoD':
        wavefxn = MDRNNWavefunction(Lx,Ly,V,Omega,delta,num_hidden,1e-3,weight_sharing,trunc,seed,cell=MDRNNGRUcell)
    else:
        raise ValueError(f"{rnn_type} is not a valid option for the RNN wave function. Please choose OneD or TwoD.")

    # ---- General Training Parameters -------------------------------------------------------------
    ns = config['ns']
    data_epochs = config.get('data_epochs',0)
    vmc_epochs = config.get('vmc_epochs',10000)
    ckpt_every = config.get('ckpt_every',10)

    if config['Print'] ==True:
        if data_epochs > 0:
            if vmc_epochs > 0:
                print(f"Training this RNN wavefunction using data-enhanced vmc.")
            else: #vmc_epochs == 0:
                print("Training this RNN wavefunction using data only.")
        else: #data_epochs == 0:
            print(f"Training this RNN wavefunction using VMC only.")

    # ---- Data Training Parameters -------------------------------------------------------------
    if data_epochs > 0:
        batch_size_data = config.get('batch_size_data', 100)
        data_lr = config.get('data_lr',1e-3)
        qmc_data = config.get('QMC_data', False)
        if qmc_data:
            dset_size = config.get('QMC_dset_size',1000)
            data = load_KZ_QMC_uncorr_data_from_batches(delta,dset_size)
            tf_dataset = create_KZ_QMC_tf_dataset(data)
            if config['Print']:
                print(f"Training on {np.shape(data)} QMC samples for {data_epochs} data-driven training steps.")
        else:
            data = data_given_param(sweep_rate,delta,Lx)
            tf_dataset = create_KZ_tf_dataset(data)   
            if config['Print']:
                print(f"Training on {np.shape(data)} experimental samples for {data_epochs} data-driven training steps.") 

    # ---- VMC Training Parameters -------------------------------------------------------------
    vmc_lr = config.get('vmc_lr',1e-3)

    # ---- Save Paths ---------------------------------------------------------------------------
    exp_name = config['name']
    scratch_path = '/scratch/msmoss/RNN_sims'
    base_path = scratch_path + f'/Rydbergs/N_{Lx*Ly}/{exp_name}/{rnn_type}_rnn/delta_{delta}/seed_{seed}'

    if data_epochs > 0:
        if qmc_data:
            data_path = base_path + '/QMC_data/dset_size_{dset_size}'
        else:
            data_path = base_path + '/Exp_data'
        write_config(config,data_path)

        hybrid_path_base = base_path + f'/hybrid_train'
        write_config(config,hybrid_path_base)
        vmc_start = load_vmc_start(hybrid_path_base,ckpt_every)
        if vmc_start == 0:
            hybrid_path = hybrid_path_base
        else:
            hybrid_path = hybrid_path_base + f'/{vmc_start}_ds/lr_{vmc_lr}'

    vmc_path = base_path + '/vmc_only'
    write_config(config,vmc_path)

    # ---- Define Train Step --------------------------------------------------------------------
    interaction_list = buildlattice(Lx,Ly,trunc)
    O_mat,V_mat,coeffs = construct_mats(interaction_list, Lx*Ly)
    Ryd_Energy_Function = get_Rydberg_Energy_Vectorized(interaction_list,wavefxn.logpsi)
    Omega_tf = tf.constant(Omega)
    delta_tf = tf.constant(delta)
    V0_tf = tf.constant(V)

    if data_epochs > 0:    
        @tf.function
        def train_step_data(input_batch):
            print("Tracing data-driven train step!")
            with tf.GradientTape() as tape:
                logpsi = wavefxn.logpsi(input_batch)
                kl_loss = - 2.0 * tf.reduce_mean(logpsi)
            gradients = tape.gradient(kl_loss, wavefxn.trainable_variables)
            clipped_gradients = [tf.clip_by_value(g, -10., 10.) for g in gradients]
            wavefxn.optimizer.apply_gradients(zip(clipped_gradients, wavefxn.trainable_variables))
            return kl_loss
    
    @tf.function
    def train_step_VMC(training_samples):
        print("Tracing Hamiltonian-driven train step!")
        with tf.GradientTape() as tape:
            training_sample_logpsi = wavefxn.logpsi(training_samples)
            with tape.stop_recording():
                training_sample_eloc = Ryd_Energy_Function(Omega_tf,delta_tf,V0_tf,O_mat,V_mat,coeffs,training_samples,training_sample_logpsi)
                sample_Eo = tf.reduce_mean(training_sample_eloc)
            energy_loss = tf.reduce_mean(2.0*tf.multiply(training_sample_logpsi, tf.stop_gradient(training_sample_eloc)) - 2.0*tf.stop_gradient(sample_Eo)*training_sample_logpsi)
            gradients = tape.gradient(energy_loss, wavefxn.trainable_variables)
            wavefxn.optimizer.apply_gradients(zip(gradients, wavefxn.trainable_variables))
        return energy_loss

    # ---- Start From CKPT or Scratch -------------------------------------------------------------
    global_step = tf.Variable(0, name="global_step")
    ckpt = tf.train.Checkpoint(step=global_step, optimizer=wavefxn.optimizer, variables=wavefxn.trainable_variables)
    if data_epochs > 0:
        data_manager = tf.train.CheckpointManager(ckpt, data_path, max_to_keep=None) # will keep checkpoints for every step
        hybrid_manager = tf.train.CheckpointManager(ckpt, hybrid_path, max_to_keep=1)
    vmc_manager = tf.train.CheckpointManager(ckpt, vmc_path, max_to_keep=1)

    if config['CKPT']:
        if data_epochs > 0:
            if (len(data_manager.checkpoints) < (data_epochs//ckpt_every - 1)):       # Finish data-driven training
                ckpt.restore(data_manager.latest_checkpoint)
                if data_manager.latest_checkpoint:
                    print("CKPT ON and ckpt found.")
                    print(f"Restored from {data_manager.latest_checkpoint}")
                    latest_ckpt = ckpt.step.numpy()
                    print(f"Continuing at step {latest_ckpt}.")
                    optimizer_initializer(wavefxn.optimizer)
                    wavefxn.optimizer.lr = data_lr
                    energy = np.load(data_path+'/Energy.npy').tolist()[0:latest_ckpt]
                    variance = np.load(data_path+'/Variance.npy').tolist()[0:latest_ckpt]
                    cost = np.load(data_path+'/Cost.npy').tolist()[0:latest_ckpt]
                else:
                    print("CKPT ON but no ckpt found. Initializing from scratch.")
                    latest_ckpt = 0
                    optimizer_initializer(wavefxn.optimizer)
                    wavefxn.optimizer.lr = data_lr
                    energy = []
                    variance = []
                    cost = []
            else: #(len(data_manager.checkpoints) >= data_epochs//ckpt_every):    # Finish hybrid training
                # Try restarting from last Hamiltonian-driven step
                ckpt.restore(hybrid_manager.latest_checkpoint)
                if hybrid_manager.latest_checkpoint:                   
                    print("CKPT ON and ckpt found.")
                    print(f"Restored from {hybrid_manager.latest_checkpoint}")
                    latest_ckpt = ckpt.step.numpy()
                    print(f"Continuing at step {latest_ckpt}.")
                    optimizer_initializer(wavefxn.optimizer)
                    wavefxn.optimizer.lr = vmc_lr
                    energy = np.load(hybrid_path+'/Energy.npy').tolist()[0:latest_ckpt]
                    variance = np.load(hybrid_path+'/Variance.npy').tolist()[0:latest_ckpt]
                    cost = np.load(hybrid_path+'/Cost.npy').tolist()[0:latest_ckpt]
                    data_epochs = 0
                # Try restarting from last data-driven step
                else:
                    ckpt.restore(data_manager.latest_checkpoint)
                    if data_manager.latest_checkpoint:
                        print("CKPT ON and ckpt found.")
                        print(f"Restored from {data_manager.latest_checkpoint}")
                        latest_ckpt = ckpt.step.numpy()
                        print(f"Continuing at step {latest_ckpt}.")
                        optimizer_initializer(wavefxn.optimizer)
                        wavefxn.optimizer.lr = data_lr
                        energy = np.load(data_path+'/Energy.npy').tolist()[0:latest_ckpt]
                        variance = np.load(data_path+'/Variance.npy').tolist()[0:latest_ckpt]
                        cost = np.load(data_path+'/Cost.npy').tolist()[0:latest_ckpt]
                    else:
                        print("CKPT ON but no ckpt found. Initializing from scratch.")
                        latest_ckpt = 0
                        optimizer_initializer(wavefxn.optimizer)
                        wavefxn.optimizer.lr = data_lr
                        energy = []
                        variance = []
                    cost = []
        elif (data_epochs == 0):                                                                # Finish vmc only training
            ckpt.restore(vmc_manager.latest_checkpoint)
            if vmc_manager.latest_checkpoint:
                print("CKPT ON and ckpt found.")
                print(f"Restored from {vmc_manager.latest_checkpoint}")
                latest_ckpt = ckpt.step.numpy()
                print(f"Continuing at step {latest_ckpt}.")
                optimizer_initializer(wavefxn.optimizer)
                wavefxn.optimizer.lr = vmc_lr
                energy = np.load(vmc_path+'/Energy.npy').tolist()[0:latest_ckpt]
                variance = np.load(vmc_path+'/Variance.npy').tolist()[0:latest_ckpt]
                cost = np.load(vmc_path+'/Cost.npy').tolist()[0:latest_ckpt]
            else:
                print("CKPT ON but no ckpt found. Initializing from scratch.")
                latest_ckpt = 0
                optimizer_initializer(wavefxn.optimizer)
                wavefxn.optimizer.lr = vmc_lr
                energy = []
                variance = []
                cost = []
        else:
            raise ValueError("data_epochs must be >= 0.")
    else:
        if data_epochs > 0:
            print("CKPT OFF. Initializing from scratch.")
            latest_ckpt = 0
            optimizer_initializer(wavefxn.optimizer)
            wavefxn.optimizer.lr = data_lr
            energy = []
            variance = []
            cost = []
        elif (data_epochs == 0):                                                                # Finish vmc only training
            print("CKPT OFF. Initializing from scratch.")
            latest_ckpt = 0
            optimizer_initializer(wavefxn.optimizer)
            wavefxn.optimizer.lr = vmc_lr
            energy = []
            variance = []
            cost = []
        else:
            raise ValueError("data_epochs must be >= 0.")
        
    # ---- Train w/ Data (?) ----------------------------------------------------------------------------------
    it = global_step.numpy()

    if data_epochs > 0:
        for n in range(it+1, data_epochs+1):
            dset = tf_dataset.shuffle(len(tf_dataset))
            dset = dset.batch(batch_size_data)
            loss = []
            for i, batch in enumerate(dset):
                batch_loss = train_step_data(batch)
                loss.append(batch_loss)
            avg_loss = np.mean(loss)
            samples, _ = wavefxn.sample(ns)
            samples_logpsi = wavefxn.logpsi(samples)
            samples_elocs = Ryd_Energy_Function(Omega_tf,delta_tf,V0_tf,O_mat,V_mat,coeffs,samples,samples_logpsi)
            avg_E = np.mean(samples_elocs.numpy())/float(wavefxn.N)
            var_E = np.var(samples_elocs.numpy())/float(wavefxn.N)
            energy.append(avg_E)
            variance.append(var_E)
            cost.append(avg_loss)
            global_step.assign_add(1)
            if (config['Print']) & (n%50 == 0):
                print(f"Step #{n}")
                print(f"Energy = {avg_E}")
                print(f"Variance = {var_E}")
                print(" ")
            if (config['CKPT']) & (n%ckpt_every == 0): # checkpoint frequently during data training
                data_manager.save()
                print(f"Saved checkpoint for step {n} in {data_path}.")
            if (config['Write_Data']) & (n%ckpt_every == 0): # save training quantities each time we checkpoint
                print(f"Saved training quantitites for step {n} in {data_path}.")
                np.save(data_path+'/Energy',energy)
                np.save(data_path+'/Variance',variance)
                np.save(data_path+'/Cost',cost)
        if config['Print']:
            print(f"Done with {data_epochs} data-driven training steps!")
    
        if (vmc_epochs > 0):
            loc_ma_energies = min_moving_average(energy,50)[0][0]
            np.save(hybrid_path+'/loc_ma_energy',loc_ma_energies)
            vmc_start = (loc_ma_energies//ckpt_every) * ckpt_every
            hybrid_path = hybrid_path_base + f'/{vmc_start}_ds/lr_{vmc_lr}'
            write_config(config,hybrid_path)
            hybrid_manager = tf.train.CheckpointManager(ckpt, hybrid_path, max_to_keep=1)
            np.save(hybrid_path+'/vmc_start_ckpt',vmc_start//ckpt_every)
            if (vmc_start > 0) & (vmc_start < len(data_manager.checkpoints)):
                ckpt.restore(data_manager.checkpoints[vmc_start])
                print(f"CKPT ON and ckpt {vmc_start} found.")
                print(f"Restored from {data_manager.checkpoints[vmc_start//ckpt_every]}")
                ckpt_step = ckpt.step.numpy()
                print(f"Continuing at step {ckpt.step.numpy()}")
                optimizer_initializer(wavefxn.optimizer)
                wavefxn.optimizer.lr = vmc_lr
                energy = np.load(data_path+'/Energy.npy').tolist()[0:ckpt_step]
                variance = np.load(data_path+'/Variance.npy').tolist()[0:ckpt_step]
                cost = np.load(data_path+'/Cost.npy').tolist()[0:ckpt_step]

    # ---- Train w/ VMC ----------------------------------------------------------------------------------
    it_vmc = global_step.numpy()
    print(f"Beginning VMC training from step {it_vmc}")
    
    for n in range(it_vmc+1, vmc_epochs+1):
        samples, _ = wavefxn.sample(ns)
        loss = train_step_VMC(samples)
        samples, _ = wavefxn.sample(ns)
        samples_logpsi = wavefxn.logpsi(samples)
        samples_elocs = Ryd_Energy_Function(Omega_tf,delta_tf,V0_tf,O_mat,V_mat,coeffs,samples,samples_logpsi)
        avg_E = np.mean(samples_elocs.numpy())/float(wavefxn.N)
        var_E = np.var(samples_elocs.numpy())/float(wavefxn.N)
        energy.append(avg_E)
        variance.append(var_E)
        cost.append(loss)
        global_step.assign_add(1)
        if (config['Print']) & (n%50 == 0):
            print(f"Step #{n}")
            print(f"Energy = {avg_E}")
            print(f"Variance = {var_E}")
            print(" ")
        if (config['CKPT']) & (n%ckpt_every == 0): # checkpoint frequently during data training
            if data_epochs > 0:
                hybrid_manager.save()
                print(f"Saved checkpoint for step {n} in {hybrid_path}.")
            else:
                vmc_manager.save()
                print(f"Saved checkpoint for step {n} in {vmc_path}.")
        if (config['Write_Data']) & (n%ckpt_every == 0): # save training quantities each time we checkpoint
            if data_epochs > 0:
                print(f"Saved training quantitites for step {n} in {hybrid_path}.")
                np.save(hybrid_path+'/Energy',energy)
                np.save(hybrid_path+'/Variance',variance)
                np.save(hybrid_path+'/Cost',cost)
            else:
                print(f"Saved training quantitites for step {n} in {vmc_path}.")
                np.save(vmc_path+'/Energy',energy)
                np.save(vmc_path+'/Variance',variance)
                np.save(vmc_path+'/Cost',cost)

    return wavefxn, energy, variance, cost
