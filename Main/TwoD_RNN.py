import numpy as np
import tensorflow as tf
from Tensordot2 import tensordot
import random

def coord_to_site_bravais(Lx, x, y):
    return Lx * y + x

class MDTensorizedRNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """The 2D Tensorized RNN cell.
    """
    def __init__(self, units = None, output_size=None, activation = None, name=None, dtype = None):
        super(MDTensorizedRNNCell, self).__init__(name=name)
        self._num_in = output_size
        self._num_units = units
        self._state_size = units
        self._output_size = output_size
        self.activation = activation

        if name != None:
            self.W = tf.Variable(name="W_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([units, 2 * units, 2 * self._num_in]),
                             dtype = dtype)

            self.b = tf.Variable(name="b_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([units]),
                             dtype = dtype)
        else:
            self.W = tf.Variable(name="W",
                             initial_value=tf.keras.initializers.GlorotNormal()([units, 2 * units, 2 * self._num_in]),
                             dtype = dtype)

            self.b = tf.Variable(name="b",
                             initial_value=tf.keras.initializers.GlorotNormal()([units]),
                             dtype = dtype)

    @property
    def input_size(self):
        return self._num_in # real

    @property
    def state_size(self):
        return self._state_size # real

    @property
    def output_size(self):
        return self._output_size # real

    @property
    def trainable_variables(self):
        return [self.W, self.b]   

    def call(self, inputs, states):

        inputstate_mul = tf.einsum('ij,ik->ijk', tf.concat(states, 1), tf.concat(inputs,1))

        state_mul = tensordot(tf, inputstate_mul, self.W, axes=[[1,2],[1,2]]) # [batch_sz, units]

        preact = state_mul + self.b

        output = self.activation(preact) # [batch_sz, units] C

        new_state = output

        return output, new_state


class MDRNNGRUcell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """
    An implementation of a 2D tensorized GRU RNN cell
    """

    def __init__(self, units=None, output_size=None, activation=None, name=None, dtype=None):
        super(MDRNNGRUcell, self).__init__(name=name)
        self._num_in = output_size
        self._num_units = units
        self._state_size = units
        self._output_size = output_size
        self.activation = activation

        self.W = tf.compat.v1.get_variable("W_" + name, shape=[units, 2 * units, 2 * self.input_size],
                                           initializer=tf.keras.initializers.VarianceScaling(scale=1.0,
                                                                                             mode="fan_avg",
                                                                                             distribution="uniform"),
                                           dtype=dtype)

        self.b = tf.compat.v1.get_variable("b_" + name, shape=[units],
                                           initializer=tf.keras.initializers.VarianceScaling(scale=1.0,
                                                                                             mode="fan_avg",
                                                                                             distribution="uniform"),
                                           dtype=dtype)

        self.Wg = tf.compat.v1.get_variable("Wg_" + name, shape=[units, 2 * units, 2 * self.input_size],
                                            initializer=tf.keras.initializers.VarianceScaling(scale=1.0,
                                                                                              mode="fan_avg",
                                                                                              distribution="uniform"),
                                            dtype=dtype)

        self.bg = tf.compat.v1.get_variable("bg_" + name, shape=[units],
                                            initializer=tf.keras.initializers.VarianceScaling(scale=1.0,
                                                                                              mode="fan_avg",
                                                                                              distribution="uniform"),
                                            dtype=dtype)

        self.Wmerge = tf.compat.v1.get_variable("Wmerge_" + name, shape=[2 * units, units],
                                                initializer=tf.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                  mode="fan_avg",
                                                                                                  distribution="uniform"),
                                                dtype=dtype)

    @property
    def trainable_variables(self):
        return [self.W, self.b, self.Wg, self.bg, self.Wmerge]

    @property
    def input_size(self):
        return self._num_in  # real

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, states):
        inputstate_mul = tf.einsum('ij,ik->ijk', tf.concat((states[0], states[1]), 1),
                                   tf.concat((inputs[0], inputs[1]), 1))
        state_mul = tensordot(tf, inputstate_mul, self.W, axes=[[1, 2], [1, 2]])  # [batch_sz, num_units]
        state_mulg = tensordot(tf, inputstate_mul, self.Wg, axes=[[1, 2], [1, 2]])  # [batch_sz, num_units]

        u = tf.nn.sigmoid(state_mulg + self.bg)
        state_tilda = tf.nn.tanh(state_mul + self.b)  # [batch_sz, num_units] C

        new_state = u * state_tilda + (1. - u) * tf.matmul(tf.concat(states, 1), self.Wmerge)
        output = new_state
        return output, new_state


class MDRNNWavefunction(object):
    def __init__(self, Lx:int, Ly:int,                #system size parameters
                 V, Omega, delta,                     #experiment parameters
                 num_hidden:int,                      #num_hidden = units!!!! = number of hidden units between RNN Cells
                 learning_rate,                       #does not get used here
                 weight_sharing = True,               #indicates whether RNN cells' weights are shared (same w dense layer)
                 trunc=2,                             #defines the order of NN interactions to include
                 seed=1234,
                 cell=MDTensorizedRNNCell):
        
        """ PARAMETERS """
        self.Lx       = Lx              # Size along x
        self.Ly       = Ly              # Size along y
        self.N        = self.Lx * self.Ly 
        self.V        = V               # Van der Waals potential
        self.Omega    = Omega           # Rabi frequency
        self.delta    = delta           # Detuning
        self.trunc    = trunc           # Truncation, set to Lx+Ly for none, default is 2
        self.nh       = num_hidden      # Number of hidden units in the RNN
        self.seed     = seed            # Seed of random number generator 
        self.K        = 2               # Dimension of the local Hilbert space
        self.weight_sharing = weight_sharing # Option to share weights between RNN cells or not (default = True)
        self.lr = learning_rate
        self.optimizer = tf.optimizers.Adam(self.lr, epsilon=1e-8)

        # Set the seed of the rng
        tf.random.set_seed(self.seed)
        
        if self.weight_sharing == True:
            self.rnn = cell(units=self.nh, output_size=self.K,activation=tf.nn.relu, name="RNN", dtype=tf.float32)
            self.dense = tf.keras.layers.Dense(self.K, activation=tf.nn.softmax, name = "Dense", dtype=tf.float32)
        else:
            self.rnn = [cell(units=self.nh, output_size=self.K,activation=tf.nn.relu, name=f"RNN_{0}_{i}", dtype=tf.float32) for i in
                            range(self.N)]
            self.dense = [tf.keras.layers.Dense(self.K, activation=tf.nn.softmax, name=f'RNNWF_dense_{i}', dtype=tf.float32) for
                            i in range(self.N)]

        # Generate trainable variables
        self.sample(1,initial_pass=False)
        self.trainable_variables = []
        if self.weight_sharing == True:
            self.trainable_variables.extend(self.rnn.trainable_variables)
            self.trainable_variables.extend(self.dense.trainable_variables)
            self.variables_names = [v.name for v in self.trainable_variables]
            sum = 0
            for k, v in zip(self.variables_names, self.trainable_variables):
                v1 = tf.reshape(v, [-1])
                sum += v1.shape[0]
            print(f'The sum of params is {sum}')
        else:
            for cell in self.rnn:
                self.trainable_variables.extend(cell.trainable_variables)
            for node_dense in self.dense:
                self.trainable_variables.extend(node_dense.trainable_variables)
            self.variables_names = [v.name for v in self.trainable_variables]
            sum = 0
            for k, v in zip(self.variables_names, self.trainable_variables):
                v1 = tf.reshape(v, [-1])
                sum += v1.shape[0]
            print(f'The total number of parameters is {sum}')

    def sample(self, numsamples,initial_pass=False):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:
            numsamples:      int
                             number of samples to be produced
            ------------------------------------------------------------------------
            Returns:         a tuple (samples,log-probs)
            samples:         tf.Tensor of shape (numsamples,systemsize_x, systemsize_y)
                             the samples in integer encoding
            log-probs        tf.Tensor of shape (numsamples,)
                             the log-probability of each sample
        """

        samples = [[[] for nx in range(self.Lx)] for ny in range(self.Ly)]
        probs = [[[] for nx in range(self.Lx)] for ny in range(self.Ly)]
        rnn_states = {}
        inputs = {}
        if initial_pass:
            print("Track which samples are drawn from which cell:")
            xs = [[[] for ny in range(self.Ly)] for nx in range(self.Ly)]
            ys = [[[] for ny in range(self.Ly)] for nx in range(self.Ly)]
            site_numbers = [[[] for ny in range(self.Ly)] for nx in range(self.Lx)]

        for ny in range(self.Ly):  # Loop over the boundaries for initialization
            if ny % 2 == 0:
                nx = -1
                inputs[f"{nx}{ny}"] = tf.zeros((numsamples, self.K),
                                               dtype=tf.float32)  # Feed the table b in tf.
                if self.weight_sharing == True:
                    rnn_states[f"{nx}{ny}"] = self.rnn.get_initial_state(inputs[f"{nx}{ny}"], dtype=tf.float32)
                else:
                    rnn_states[f"{nx}{ny}"] = self.rnn[0].get_initial_state(inputs[f"{nx}{ny}"], dtype=tf.float32)

            if ny % 2 == 1:
                nx = self.Lx
                inputs[f"{nx}{ny}"] = tf.zeros((numsamples, self.K),
                                               dtype=tf.float32)  # Feed the table b in tf.
                if self.weight_sharing == True:
                    rnn_states[f"{nx}{ny}"] = self.rnn.get_initial_state(inputs[f"{nx}{ny}"], dtype=tf.float32)
                else:
                    rnn_states[f"{nx}{ny}"] = self.rnn[0].get_initial_state(inputs[f"{nx}{ny}"], dtype=tf.float32)
                
        for nx in range(self.Lx):  # Loop over the boundaries for initialization
            ny = -1
            inputs[f"{nx}{ny}"] = tf.zeros((numsamples, self.K),
                                           dtype=tf.float32)  # Feed the table b in tf.
            if self.weight_sharing == True:
                rnn_states[f"{nx}{ny}"] = self.rnn.get_initial_state(inputs[f"{nx}{ny}"], dtype=tf.float32)
            else:
                rnn_states[f"{nx}{ny}"] = self.rnn[0].get_initial_state(inputs[f"{nx}{ny}"], dtype=tf.float32)
        
        # Making a loop over the sites with the 2DRNN
        for ny in range(self.Ly):

            if ny % 2 == 0:

                for nx in range(self.Lx):  # left to right
                    
                    if self.weight_sharing == True:
                        rnn_output,rnn_states[f"{nx}{ny}"] = self.rnn((inputs[f"{nx - 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx - 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense(rnn_output)
                        
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[ny * self.Lx + nx]((inputs[f"{nx - 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx - 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense[ny * self.Lx + nx](rnn_output)

                    sample_temp = tf.reshape(tf.random.categorical(tf.math.log(output), num_samples=1), [-1, ])
                    samples[nx][ny] = sample_temp
                    probs[nx][ny] = output
                    inputs[f"{nx}{ny}"] = tf.one_hot(sample_temp, depth=self.K, dtype=tf.float32)

                    if initial_pass:
                        xs[nx][ny] = nx
                        ys[nx][ny] = ny
                        site_numbers[nx][ny] = coord_to_site_bravais(self.Lx,nx,ny)
                        print(f"at {nx},{ny} (site {coord_to_site_bravais(self.Lx,nx,ny)}), the sample is {sample_temp}")

            if ny % 2 == 1:

                for nx in range(self.Lx - 1, -1, -1):  # right to left

                    if self.weight_sharing == True:
                        rnn_output,rnn_states[f"{nx}{ny}"] = self.rnn((inputs[f"{nx + 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx + 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense(rnn_output)
                        
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[ny * self.Lx + nx]((inputs[f"{nx + 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx + 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense[ny * self.Lx + nx](rnn_output)
                    
                    sample_temp = tf.reshape(tf.random.categorical(tf.math.log(output), num_samples=1), [-1, ])
                    samples[nx][ny] = sample_temp
                    probs[nx][ny] = output
                    inputs[f"{nx}{ny}"] = tf.one_hot(sample_temp, depth=self.K, dtype=tf.float32)

                    if initial_pass:
                        xs[nx][ny] = nx
                        ys[nx][ny] = ny
                        site_numbers[nx][ny] = coord_to_site_bravais(self.Lx,nx,ny)
                        print(f"at {nx},{ny} (site {coord_to_site_bravais(self.Lx,nx,ny)}), the sample is {sample_temp}")

        samples = tf.stack(values=samples, axis=1)
        samples = tf.transpose(samples, perm=[2, 0, 1])
        probs = tf.transpose(tf.stack(values=probs, axis=0), perm=[2, 0, 1, 3])
        one_hot_samples = tf.one_hot(samples, depth=self.K, dtype=tf.float32)
        log_probs = 0.5 * tf.reduce_sum(tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs, one_hot_samples), axis=3)), axis=2), axis=1)
        full_samples = tf.reshape(samples,(numsamples,self.Lx*self.Ly)) 
        if initial_pass:
            xs = tf.stack(values=xs,axis=1)
            ys = tf.stack(values=ys,axis=1)
            site_numbers = tf.stack(values=site_numbers,axis=1)

        if initial_pass:
            return xs, ys, site_numbers, samples, full_samples
        else:
            return full_samples, log_probs

    def logpsi(self, samples,initial_pass=False):
        """
        calculate the log-probabilities of ```samples``
        ------------------------------------------------------------------------
        Parameters:
        samples:         tf.Tensor
                         a tf.placeholder of shape (number of samples,systemsize_x, systemsize_y)
                         containing the input samples in integer encoding
        ------------------------------------------------------------------------
        Returns:
        log-probs        tf.Tensor of shape (number of samples,)
                         the log-probability of each sample
        """
        samples_reconstructed = [[[] for nx in range(self.Lx)] for ny in range(self.Ly)]
        probs = [[[] for nx in range(self.Lx)] for ny in range(self.Ly)]
        numsamples = samples.shape[0]
        samples_ = tf.reshape(samples, (numsamples,self.Lx,self.Ly))
        samples_input = tf.transpose(samples_, perm=[2, 1, 0])
        rnn_states = {}
        inputs = {}
        if initial_pass:
            print("\nCheck that samples get passed back through the same cell:")

        for ny in range(self.Ly):  # Loop over the boundaries for initialization
            if ny % 2 == 0:
                
                nx = -1
                if self.weight_sharing == True:
                    rnn_states[f"{nx}{ny}"] = self.rnn.zero_state(numsamples, dtype=tf.float32)
                else:
                    rnn_states[f"{nx}{ny}"] = self.rnn[0].zero_state(numsamples, dtype=tf.float32)
                inputs[f"{nx}{ny}"] = tf.zeros((numsamples, self.K),dtype=tf.float32)  # Feed the table b in tf.

            if ny % 2 == 1:
                
                nx = self.Lx
                if self.weight_sharing == True:
                    rnn_states[f"{nx}{ny}"] = self.rnn.zero_state(numsamples, dtype=tf.float32)
                else:
                    rnn_states[f"{nx}{ny}"] = self.rnn[0].zero_state(numsamples, dtype=tf.float32)
                inputs[f"{nx}{ny}"] = tf.zeros((numsamples, self.K),dtype=tf.float32)  # Feed the table b in tf.

        for nx in range(self.Lx):  # Loop over the boundaries for initialization
            ny = -1
            if self.weight_sharing == True:
                rnn_states[f"{nx}{ny}"] = self.rnn.zero_state(numsamples, dtype=tf.float32)
            else:
                rnn_states[f"{nx}{ny}"] = self.rnn[0].zero_state(numsamples, dtype=tf.float32)
            inputs[f"{nx}{ny}"] = tf.zeros((numsamples, self.K),dtype=tf.float32)  # Feed the table b in tf.

        # Making a loop over the sites with the 2DRNN
        for ny in range(self.Ly):

            if ny % 2 == 0:

                for nx in range(self.Lx):  # left to right

                    if self.weight_sharing == True:
                        rnn_output,rnn_states[f"{nx}{ny}"] = self.rnn((inputs[f"{nx - 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx - 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense(rnn_output)
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[ny * self.Lx + nx]((inputs[f"{nx - 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx - 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense[ny * self.Lx + nx](rnn_output)

                    probs[nx][ny] = output
                    sample_temp = samples_input[nx, ny]
                    samples_reconstructed[nx][ny] = sample_temp
                    inputs[f"{nx}{ny}"] = tf.one_hot(sample_temp, depth=self.K, dtype=tf.float32)
                    if initial_pass:
                        print(f"at {nx},{ny} (site {coord_to_site_bravais(self.Lx,nx,ny)}) the input sample is {samples_input[nx, ny]}.")
                        print(f"the log probs are:\n{output}")

            if ny % 2 == 1:

                for nx in range(self.Lx - 1, -1, -1):  # right to left

                    if self.weight_sharing == True:
                        rnn_output,rnn_states[f"{nx}{ny}"] = self.rnn((inputs[f"{nx + 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx + 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense(rnn_output)
                    else:
                        rnn_output, rnn_states[f"{nx}{ny}"] = self.rnn[ny * self.Lx + nx]((inputs[f"{nx + 1}{ny}"],inputs[f"{nx}{ny - 1}"]),(rnn_states[f"{nx + 1}{ny}"],rnn_states[f"{nx}{ny - 1}"]))
                        output = self.dense[ny * self.Lx + nx](rnn_output)
                        
                    probs[nx][ny] = output
                    sample_temp = samples_input[nx, ny]
                    samples_reconstructed[nx][ny] = sample_temp
                    inputs[f"{nx}{ny}"] = tf.one_hot(sample_temp, depth=self.K, dtype=tf.float32)
                    if initial_pass:
                        print(f"at {nx},{ny} (site {coord_to_site_bravais(self.Lx,nx,ny)}) the input sample is {samples_input[nx, ny]}.")
                        print(f"the log probs are:\n{output}")

        probs = tf.stack(values=probs, axis=1)
        probs = tf.transpose(probs, perm=[2, 0, 1, 3])
        samples_reconstructed = tf.stack(values=samples_reconstructed,axis=1)
        samples_reconstructed = tf.transpose(samples_reconstructed,perm=[2,0,1])
        if initial_pass:
            assert(np.all(samples_reconstructed.numpy()==samples_.numpy()))
        one_hot_samples = tf.one_hot(samples_, depth=self.K, dtype=tf.float32)
        log_probs = 0.5 * tf.reduce_sum(tf.reduce_sum(tf.math.log(tf.clip_by_value(tf.reduce_sum(tf.multiply(probs, one_hot_samples), axis=3), 1e-10, 1.0)),axis=2), axis=1)

        return log_probs
