
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from IPython import display
from IPython.display import clear_output

from os import listdir
from os.path import isfile, join

import numpy as np
import keras
import scipy.io
from scipy.io import loadmat,savemat
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

#-------------------- This are main py------------------------
from Read_Spectral import *   # data set build
from recoverynet import *     # Net build

# To run in cpu
#os.environ["CUDA_VISIBLE_DEVICES"]= '-1'
os.environ["CUDA_VISIBLE_DEVICES"]= '0'




#----------------------------- directory of the spectral data set -------------------------
#PATH = '/media/hdsp-deep/A2CC8AC9CC8A96E7/Spectral_data_set/data_500_spta_512_band_24/' # Carga de datos   # for linux
#PATH = r'E:\Hans_pivado\Deep_learning_aprendizaje\Training_method_keras_model\Training_method_keras_model\Data_set'                                # for windows
PATH = r'C:\Cavings_proyect\Datos_Caving_Julio_23\Modelo_volumen\Data_train'                               # for windows

# parameters of the net
BATCH_SIZE = 3; IMG_WIDTH = 1080; IMG_HEIGHT = 1920; L_bands    = 1; L_imput    = 2; split_v = 0.80 # L_bands es las bandas de la salida

test_dataset,train_dataset=Build_data_set(IMG_WIDTH,IMG_HEIGHT,L_bands,L_imput,BATCH_SIZE,PATH,split_v)  # build the DataStore from Read_Spectral.py

#-------------Net_model----------------------------------------------------------------
model = UNetL(input_size=(IMG_HEIGHT,IMG_WIDTH,L_imput))


optimizad = tf.keras.optimizers.Adam(learning_rate=1e-4, amsgrad=False)
model.compile(optimizer=optimizad, loss='mean_squared_error')
model.summary()
model.load_weights('./checkpoints/Vol_200.tf')



# -------------- Tratando de armar la mascara ------------------------
#L=16
#N=128
#M=128
#Nt=64
#####wave_lengths = np.linspace(420, 660, L)*1e-9
#####fr, fg, fc, fb = get_color_bases(wave_lengths)
#####
#####WR=model.layers[1].get_weights()[0]
#####WG=model.layers[1].get_weights()[1]
#####WB=model.layers[1].get_weights()[2]
#####WC=model.layers[1].get_weights()[3]
#####
#####WT = WR + WG + WB + WC
#####wr = tf.math.divide(WR, WT)
#####wg = tf.math.divide(WG, WT)
#####wb = tf.math.divide(WB, WT)
#####wc = tf.math.divide(WC, WT)
#####Aux1 = tf.multiply(wr, fr) + tf.multiply(wg, fg) + tf.multiply(wb, fb) + tf.multiply(wc, fc)
#####Mask = kronecker_product(tf.ones((int(M/Nt), int(N/Nt))), Aux1)
#####Mask = tf.expand_dims(Mask, 0)
#####
#####Mask_NP=Mask.numpy().squeeze()
#####plt.subplot(131)
#####plt.imshow(Mask_NP[:,:,[15,8,3]])
#####plt.show()
## See some reconstruction
Img_spectral = loadmat(r'C:\Cavings_proyect\Datos_Caving_Julio_23\Modelo_volumen\Data_train\Data_2_1017.mat')
#Img_spectral = loadmat(r'C:\Cavings_proyect\Datos_Caving_Julio_23\Modelo_volumen\Data_train\Data_4_1027.mat')
Ref_img = Img_spectral['Dato']


Ref_img2=np.zeros((1920,1080,2))
Aux=np.squeeze(Ref_img[:,:,[0]])
Ref_img2[:,:,[0]]=np.expand_dims(np.transpose(Aux),2)
Aux=np.squeeze(Ref_img[:,:,[1]])
Ref_img2[:,:,[1]]=np.expand_dims(np.transpose(Aux),2)
Ref_img2=np.expand_dims(Ref_img2,0)

Resul= model.predict(Ref_img2,batch_size=1)
Resul=Resul[0,:,:,:]



temp = Ref_img2[:,:,:,[0]]/np.max( Ref_img2[:,:,:,[0]])
plt.subplot(141)
plt.imshow(np.squeeze(temp))
#plt.show()


temp1 = Ref_img2[:,:,:,[1]]/np.max( Ref_img2[:,:,:,[1]])
plt.subplot(143)
plt.imshow(np.squeeze(temp1))
#plt.show()



temp1 = Resul[:,:,[0]]/np.max( Resul[:,:,[0]])
plt.subplot(142)
plt.imshow(np.squeeze(temp1))
plt.show()

RP = np.transpose(Img_spectral['RP'])
RP = RP/np.max(RP)
plt.subplot(144)
plt.imshow(RP)
plt.show()


scipy.io.savemat("recovery.mat", {'Resul':Resul})
# medidas


