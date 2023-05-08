import os
import tensorflow as tf
import numpy as np

def optimizer_initializer(optimizer):
    fake_var = tf.Variable(1.0)
    with tf.GradientTape() as tape:
        fake_loss = tf.reduce_sum(fake_var ** 2)
    grads = tape.gradient(fake_loss, [fake_var])
    # Ask the optimizer to apply the processed gradients.
    optimizer.apply_gradients(zip(grads, [fake_var]))

def ma(values,window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def min_moving_average(values,window):
    ma_values = ma(values,window)
    min_ma = min(ma_values)
    min_loc = np.where(ma_values==min_ma)
    return min_loc

def write_config(config,path):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+'/config.txt', 'w') as file:
        for k,v in config.items():
            file.write(k+f'={v}\n')

def load_vmc_start(path,ckpt_every):
    if os.path.exists(path+'/loc_ma_energy.npy'):
        vmc_start = np.load(path+'/loc_ma_energy.npy')
        vmc_start = (vmc_start//ckpt_every) * ckpt_every
        print(vmc_start)
        if not os.path.exists(path + f"/{vmc_start}_ds"):
            vmc_start = 0
    else:
        vmc_start = 0
    return vmc_start