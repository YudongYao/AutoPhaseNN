#Keras modules
from tensorflow.keras.layers import Conv3D, MaxPool3D, UpSampling3D, ZeroPadding3D
from tensorflow.keras.layers import LeakyReLU,BatchNormalization

import numpy as np


# encoder layers
def Conv_Pool_block(x0,nfilters,w1=3,w2=3,w3=3,p1=2,p2=2,p3=2,Lalpha = 0.05, padding='same', data_format='channels_last'):
    x0 = Conv3D(nfilters, (w1, w2, w3), padding=padding, data_format=data_format)(x0)
    x0 = LeakyReLU(alpha=Lalpha)(x0)
    x0 = BatchNormalization()(x0)
      
    x0 = Conv3D(nfilters, (w1, w2, w3), padding=padding, data_format=data_format)(x0)
    x0 = LeakyReLU(alpha=Lalpha)(x0)
    x0 = BatchNormalization()(x0)
    
    x0 = MaxPool3D((p1, p2, p3), padding=padding, data_format=data_format)(x0)
    return x0
  
def Conv_Pool_block_last(x0,nfilters,w1=3,w2=3,w3=3,p1=2,p2=2,p3=2, Lalpha = 0.05, padding='same', data_format='channels_last'):
    x0 = Conv3D(nfilters, (w1, w2, w3), padding=padding, data_format=data_format)(x0)
    x0 = LeakyReLU(alpha=Lalpha)(x0)
    x0 = BatchNormalization()(x0)
    
    x0 = Conv3D(nfilters, (w1, w2, w3), padding=padding, data_format=data_format)(x0)
    x0 = LeakyReLU(alpha=Lalpha)(x0)
    x0 = BatchNormalization()(x0)
    
    return x0


# decoder layer
def Conv_Upfirst_block(x0,nfilters,w1=3,w2=3,w3=3,p1=2,p2=2,p3=2,Lalpha = 0.05, padding='same', data_format='channels_last'):
    x0 = UpSampling3D((p1, p2, p3), data_format=data_format)(x0)
    
    x0 = Conv3D(nfilters, (w1, w2, w3), padding=padding, data_format=data_format)(x0)
    x0 = LeakyReLU(alpha=Lalpha)(x0)
    x0 = BatchNormalization()(x0)
    
    x0 = Conv3D(nfilters, (w1, w2, w3), padding=padding, data_format=data_format)(x0)
    x0 = LeakyReLU(alpha=Lalpha)(x0)
    x0 = BatchNormalization()(x0)
    
    
    return x0

def Conv_Upfirst_block_last(x0,nfilters,w1=3,w2=3,w3=3,psize=16,padding='same', data_format='channels_last'):
    x0 = ZeroPadding3D(padding = psize, data_format=data_format)(x0)
    
    x0 = Conv3D(nfilters, (w1, w2, w3), activation='relu',padding=padding, data_format=data_format)(x0)
    x0 = BatchNormalization()(x0)
    
    
    x0 = Conv3D(nfilters, (w1, w2, w3), activation='relu',padding=padding, data_format=data_format)(x0)
    x0 = BatchNormalization()(x0)

    return x0

def Conv_Upfirst_block_mlast(x0,nfilters,w1=3,w2=3,w3=3,psize=16,padding='same', data_format='channels_last'):
    x0 = ZeroPadding3D(padding = psize, data_format=data_format)(x0)
    
    x0 = Conv3D(nfilters, (w1, w2, w3), activation='relu',padding=padding, data_format=data_format)(x0)
    x0 = Conv3D(nfilters, (w1, w2, w3), activation='relu',padding=padding, data_format=data_format)(x0)

    return x0

