
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from IPython import display
from IPython.display import clear_output

from os import listdir
from os.path import isfile, join

import numpy as np
from tensorflow import keras
import scipy.io
from scipy.io import loadmat
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from recovery.CASSI_layer_v2 import *


def UNetL(pretrained_weights=None, input_size=(256, 256, 12)):

    x = inputs = Input(input_size)
    #est, y = CASSI_Layer_v2(input_size, input_size[0], input_size[1], input_size[2], Nt=64,
                            #wave_lengths=np.linspace(420, 660, input_size[2]) * 1e-9)(x)
    L = 4
    L_2 = 2 * L
    L_3 = 3 * L
    L_4 = 4 * L
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    #conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)############
    #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

    bt=BatchNormalization()(conv4)

    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(bt)
    # drop4 = Dropout(0.5)(conv4)

    up5 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    #conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)#############

    #up7 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        #UpSampling2D(size=(2, 2))(conv6))
    up7 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv7)

    final = Conv2D(1, 1)(conv7)


    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model