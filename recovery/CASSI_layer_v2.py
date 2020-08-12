import tensorflow as tf  # se puede cambiar por from keras.import backend as K
from tensorflow.keras.layers import Layer  # quitar tensorflow si usa keras solo
from tensorflow.keras.constraints import NonNeg
import numpy as np
import os
from random import random
from scipy.io import loadmat
from recovery.functionsC import deta, ifftshift, area_downsampling_tf, compl_exp_tf, transp_fft2d, transp_ifft2d, img_psf_conv,fftshift2d_tf,get_color_bases,propagation,propagation_back,kronecker_product
from tensorflow.keras.constraints import NonNeg


class CASSI_Layer_v2(Layer):

    def __init__(self, output_dim, M=512, N=512, L=12, Nt=32, wave_lengths=None, **kwargs):

        self.output_dim = output_dim
        self.M = M
        self.N = N
        self.L = L
        self.Nt = Nt

        if wave_lengths is not None:
            self.wave_lengths = wave_lengths
        else:
            self.wave_lengths = np.linspace(420, 660, L)*1e-9

        self.fr, self.fg, self.fc, self.fb = get_color_bases(self.wave_lengths)

        super(CASSI_Layer_v2, self).__init__(**kwargs)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'M': self.M,
            'N': self.N,
            'L': self.L,
            'Nt': self.Nt,
            'wave_lengths': self.wave_lengths,
            'fr': self.fr,
            'fg': self.fg,
            'fb': self.fb,
            'fc': self.fc,
        })
        return config

    def build(self, input_shape):

        wr = (np.random.rand(self.Nt, self.Nt))
        wg = (np.random.rand(self.Nt, self.Nt))
        wb = (np.random.rand(self.Nt, self.Nt))
        wc = (np.random.rand(self.Nt, self.Nt))
        wt = wr + wg + wb + wc
        wr =  tf.constant_initializer(tf.math.divide(wr, wt))
        wg =  tf.constant_initializer(tf.math.divide(wg, wt))
        wb =  tf.constant_initializer(tf.math.divide(wb, wt))
        wc =  tf.constant_initializer(tf.math.divide(wc, wt))

        self.wr = self.add_weight(name='wr', shape=(self.Nt,self.Nt, 1),
                                  initializer=wr, trainable=True, constraint=NonNeg())
        self.wg = self.add_weight(name='wg', shape=(self.Nt,self.Nt, 1),
                                  initializer=wg, trainable=True, constraint=NonNeg())
        self.wb = self.add_weight(name='wb', shape=(self.Nt,self.Nt, 1),
                                  initializer=wb, trainable=True, constraint=NonNeg())
        self.wc = self.add_weight(name='wc', shape=(self.Nt, self.Nt, 1),
                                  initializer=wc, trainable=True, constraint=NonNeg())

        self.batch_size = input_shape[0]

        super(CASSI_Layer_v2, self).build(input_shape)

    def call(self, inputs, **kwargs):

        wt = self.wr + self.wg + self.wb + self.wc
        wr = tf.math.divide(self.wr, wt)
        wg = tf.math.divide(self.wg, wt)
        wb = tf.math.divide(self.wb, wt)
        wc = tf.math.divide(self.wc, wt)

        Aux1 = tf.multiply(wr, self.fr) + tf.multiply(wg, self.fg) + tf.multiply(wb, self.fb) + tf.multiply(wc, self.fc)
        Mask = kronecker_product(tf.ones((int(self.M/self.Nt), int(self.N/self.Nt))), Aux1)
        Mask = tf.expand_dims(Mask, 0)

        Images = tf.convert_to_tensor(tf.ones((1, int(self.M), int(self.N), self.L)), dtype=tf.float32)

        # CASSI Sensing Model
        Aux1 = tf.multiply(Mask, inputs)
        Aux1 = tf.pad(Aux1, [[0, 0], [0, 0], [0, self.L - 1], [0, 0]])
        Y = None
        for i in range(self.L):
            Tempo = tf.roll(Aux1, shift=i, axis=2)
            if Y is not None:
                Y = tf.concat([Y, tf.expand_dims(Tempo[:, :, :, i], -1)], axis=3)
            else:
                Y = tf.expand_dims(Tempo[:, :, :, i], -1)
        Y = tf.reduce_sum(Y, 3)
      

        # CASSI Transpose model (x = H'*y)
        X = None
        for i in range(self.L):
            Tempo = tf.roll(Y, shift=-i, axis=2)
            if X is not None:
                X = tf.concat([X, tf.expand_dims(Tempo[:, :, 0:self.M], -1)], axis=3)
            else:
                X = tf.expand_dims(Tempo[:, :, 0:self.M], -1)

        X = tf.multiply(Mask, X)
        return X, tf.expand_dims(Y,-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)