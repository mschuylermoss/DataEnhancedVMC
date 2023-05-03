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

