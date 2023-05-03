import scipy
import scipy.sparse.linalg as sspla
import numpy as np
import tensorflow as tf
from math import ceil

""" Generate the square lattice structures """
def coord_to_site(Ly,x,y):
    return Ly*x+y
    
def buildlattice(Lx,Ly,trunc:int,):
    interactions = []
        
    for n in range(1,Lx):
        for n_ in range(n+1):
            if n+n_ > trunc:
                continue
            else:
                for x in range(Lx-n_):
                    for y in range(Ly-n):
                        coeff = np.sqrt(n**2+n_**2)**6
                        if n_ == 0 :
                                interactions.append([coeff,coord_to_site(Ly,x,y),coord_to_site(Ly,x,y+n)])
                        elif n == n_: 
                            interactions.append([coeff,coord_to_site(Ly,x,y),coord_to_site(Ly,x+n,y+n)])
                            interactions.append([coeff,coord_to_site(Ly,x+n,y),coord_to_site(Ly,x,y+n)])
                        else:
                            interactions.append([coeff,coord_to_site(Ly,x,y),coord_to_site(Ly,x+n_,y+n)])
                            interactions.append([coeff,coord_to_site(Ly,x+n_,y),coord_to_site(Ly,x,y+n)])
                for y in range(Ly-n_):
                    for x in range(Lx-n):
                        coeff = np.sqrt(n**2+n_**2)**6
                        if n_ == 0 :
                            interactions.append([coeff,coord_to_site(Ly,x,y),coord_to_site(Ly,x+n,y)])
                        elif n == n_: 
                            continue #already counted above
                        else:
                            interactions.append([coeff,coord_to_site(Ly,x,y),coord_to_site(Ly,x+n,y+n_)])
                            interactions.append([coeff,coord_to_site(Ly,x,y+n_),coord_to_site(Ly,x+n,y)])

    return interactions

def construct_mats(interaction_list, n_spins):
    num_interactions = len(interaction_list)

    V_mat = np.zeros((num_interactions,n_spins))
    coeffs = np.zeros((num_interactions,1))
    for n,interaction in enumerate(interaction_list):
        coeff,i,j = interaction
        coeffs[int(n),0] = 1/coeff
        V_mat[int(n),int(i)] += 1
        V_mat[int(n),int(j)] += 1

    O_mat = np.zeros((n_spins,n_spins))
    for i in range(n_spins):
        O_mat[i,i] += 1
    
    V_mat = tf.constant(V_mat,dtype=tf.float32)
    O_mat = tf.constant(O_mat,dtype=tf.float32)
    coeffs = tf.constant(coeffs,dtype=tf.float32)

    return O_mat,V_mat,coeffs

def get_Rydberg_Energy_Vectorized(interactions, log_fxn):
    print("getting energy function!")
    @tf.function()
    def Rydberg_Energy_Vectorized(O,d,V0,O_mat,V_mat,coeffs,samples,sample_log_probs):
        numsamples = tf.shape(samples)[0]
        N = tf.shape(samples)[1]
        samples = tf.cast(samples,dtype=tf.float32)
        energies_delta = tf.reduce_sum(samples,axis=1) # sum over occupation operators, so only count the ones
        # print("shape of term 1: ",energies_delta.shape)
        ni_plus_nj = (V_mat @ tf.transpose(samples))
        # print(ni_plus_nj)
        ni_nj = tf.math.floor(ni_plus_nj/2)
        # print(ni_nj)
        # print(tf.reduce_sum(ni_nj)) # correct up to here
        c_ni_nj = tf.multiply(coeffs,ni_nj)
        # print(c_ni_nj)
        energies_V = tf.reduce_sum(c_ni_nj,axis=0) # interaction terms, done efficiently
        # print(energies_V)
        # print("shape of term 2: ",energies_V.shape)
        samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], N, axis=2)
        # print("tile samples: ",samples_tiled_not_flipped.shape)
        # print("The tiled samples are:")
        # print(samples_tiled_not_flipped)
        samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped + tf.transpose(O_mat)[tf.newaxis, :, :], 2) # numsamples, n_spins, n_spins
        # print("flip tiled samples: ",samples_tiled_flipped.shape)
        # print("The tiled and flipped samples are: ")
        # print(samples_tiled_flipped)
        samples_tiled_flipped = tf.transpose(samples_tiled_flipped,perm=[0,2,1]) # MIGHT NOT NEED THIS!
        # print("transpose... ")
        # print(samples_tiled_flipped)
        samples_tiled_flipped = tf.reshape(samples_tiled_flipped,(numsamples*N,N))
        # print("reshape... ")
        # print(samples_tiled_flipped)
        samples_tiled_flipped = tf.cast(samples_tiled_flipped,dtype=tf.int64)
        log_probs_flipped = log_fxn(samples_tiled_flipped)
        # print("compute log probs of flipped samples: ",log_probs_flipped.shape)
        # print(log_probs_flipped)
        log_probs_flipped = tf.reshape(log_probs_flipped,(numsamples,N))
        # print(log_probs_flipped)
        # print(sample_log_probs)
        # print(log_probs_flipped - sample_log_probs[:, tf.newaxis])
        log_prob_ratio = tf.math.exp(log_probs_flipped - sample_log_probs[:, tf.newaxis])
        energies_O = tf.reduce_sum(log_prob_ratio,axis=1)
        # print(energies_O.shape)
        # print(-1*energies_delta)
        # print(-1*energies_delta + V0 * energies_V)
        # print(-1 * d * energies_delta + V0 * energies_V - (1/2) * O * energies_O)
        Energies = -1 * d * energies_delta + V0 * energies_V - (1/2) * O * energies_O
        return Energies    
    return Rydberg_Energy_Vectorized

    # def Heisenberg_Energy_Vectorized_tf_function(J, J_matrix, samples, og_amps):
    #     N = tf.shape(samples)[1]
    #     Energies_zz = tf.reduce_sum(tf.math.abs(J_matrix @ (2 * tf.transpose(samples) - 1)) - 1, axis=0)
    #     Energies_zz = tf.complex(Energies_zz, 0.0)
    #     samples_tiled_not_flipped = tf.repeat(samples[:, :, tf.newaxis], len(interactions), axis=2)
    #     samples_tiled_flipped = tf.math.mod(samples_tiled_not_flipped +
    #                                             tf.transpose(J_matrix)[tf.newaxis, :, :], 2)
    #     samples_tiled_sub = samples_tiled_flipped - samples_tiled_not_flipped
    #     signs = tf.complex(tf.math.abs(tf.reduce_sum(samples_tiled_sub, axis=1)) - 1, 0.0)
    #     samples_tiled_flipped = tf.transpose(samples_tiled_flipped, perm=[0, 2, 1])
    #     flip_logprob, flip_logamp = log_fxn(tf.reshape(samples_tiled_flipped, (-1, N)), symmetrize=symmetrize,initial_pass=False)
    #     amp_ratio = tf.math.exp(tf.reshape(flip_logamp, (-1, len(interactions))) - og_amps[:, tf.newaxis])
    #     Energies_xx = tf.reduce_sum(amp_ratio, axis=1)
    #     Energies_yy = -tf.reduce_sum(signs * amp_ratio, axis=1)
    #     Energies = 0.25 * J * (ms * Energies_xx + ms * Energies_yy + Energies_zz)
    #     return Energies


    # def localenergy(self,samples,logpsi):
    #     eloc = tf.zeros(shape=[tf.shape(samples)[0]],dtype=tf.float32)
    #     # Chemical potential
    #     for j in range(self.N):
    #         eloc += - self.delta * tf.cast(samples[:,j],tf.float32)
    #     for n in range(len(self.interactions)):
    #         eloc += (self.V/self.interactions[n][0]) * tf.cast(samples[:,self.interactions[n][1]]*samples[:,self.interactions[n][2]],tf.float32)
    #     flip_logpsi = tf.zeros(shape=[tf.shape(samples)[0]])
    #     # Off-diagonal part
    #     for j in range(self.N):
    #         flip_samples = np.copy(samples)
    #         flip_samples[:,j] = 1 - flip_samples[:,j]
    #         flip_logpsi = self.logpsi(flip_samples)
    #         eloc += -0.5*self.Omega * tf.math.exp(flip_logpsi-logpsi)
    #     return eloc