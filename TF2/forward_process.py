from tensorflow.keras.layers import Layer
from math import pi as pi
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


@tf.function
def combine_complex(amp, phi):
    import tensorflow as tf
    output = tf.cast(amp, tf.complex64) * tf.exp(
        1j * tf.cast(phi, tf.complex64))
    return output


@tf.function
def ff_propagation(data):
    import tensorflow as tf
    '''
    diffraction. Assume same x and y lengthss and uniform sampling
        data:        source plane field
        
    '''
    diff = _fourier_transform(data)

    # far-field amplitude
    intensity = tf.math.abs(diff)

    intensity = tf.cast(intensity, tf.float32)

    return intensity


@tf.function
# 3D fourier transform
def _fourier_transform(input):
    import tensorflow as tf
    # fft3d transform with channel unequal to 1
    perm_input = K.permute_dimensions(input, pattern=[4, 0, 1, 2, 3])
    perm_Fr = tf.signal.fftshift(tf.signal.fft3d(
        tf.signal.ifftshift(tf.cast(perm_input, tf.complex64),
                            axes=[-3, -2, -1])),
                                 axes=[-3, -2, -1])
    Fr = K.permute_dimensions(perm_Fr, pattern=[1, 2, 3, 4, 0])

    return Fr


# @tf.function
# def get_mask(input):
#     import tensorflow as tf

#     mask = tf.where(input >= 0.1, tf.ones_like(input), tf.zeros_like(input))
#     return mask


@tf.function
def loss_log(Y_true, Y_pred):
    pred = tf.experimental.numpy.log10(Y_pred+1)
    true = tf.experimental.numpy.log10(Y_true+1)

    top = tf.reduce_sum(tf.math.square(pred - true))
    bottom = tf.reduce_sum(tf.math.square(true))
    # loss_value = tf.sqrt(top / bottom)
    loss_value = top / bottom
    return loss_value

@tf.function
def loss_sq(Y_true, Y_pred):
    top = tf.reduce_sum(tf.math.square(Y_pred - Y_true),axis=(1,2,3),keepdims=True)
    bottom = tf.reduce_sum(tf.math.square(Y_true),axis=(1,2,3),keepdims=True)

    loss_value = tf.reduce_sum(top / bottom)
    return loss_value

@tf.function
def loss_mae(Y_true, Y_pred):
    top = tf.reduce_sum(tf.math.abs(Y_pred - Y_true),axis=(1,2,3),keepdims=True)
    bottom = tf.reduce_sum(tf.math.abs(Y_true),axis=(1,2,3),keepdims=True)
    loss_value = tf.reduce_sum(top / bottom)
    return loss_value

@tf.function
def loss_pcc(Y_true, Y_pred):
    pred = Y_pred - tf.reduce_mean(Y_pred,axis=(1,2,3),keepdims=True)
    true = Y_true - tf.reduce_mean(Y_true,axis=(1,2,3),keepdims=True)
    top = tf.reduce_sum(pred * true,axis=(1,2,3),keepdims=True)
    
    pred_sum = tf.reduce_sum(pred**2,axis=(1,2,3),keepdims=True)
    true_sum = tf.reduce_sum(true**2,axis=(1,2,3),keepdims=True)
    bottom = tf.math.sqrt(pred_sum * true_sum)
    
    loss_value = tf.reduce_sum(1 - top / bottom)
    return loss_value

@tf.function
def loss_comb(Y_true, Y_pred):
    loss_1 = loss_sq(Y_true, Y_pred)
    loss_2 = loss_pcc(Y_true, Y_pred)
    a1 = 1
    a2 = 1
    loss_value = (a1*loss_1+a2*loss_2)/(a1+a2)
    return loss_value

@tf.function
def loss_comb2(Y_true, Y_pred):  
    loss_1 = tf.math.sqrt(loss_sq(Y_true, Y_pred))
    loss_2 = loss_pcc(Y_true, Y_pred)
    a1 = 1
    a2 = 1
    loss_value = (a1*loss_1+a2*loss_2)/(a1+a2)
    return loss_value

@tf.function
def loss_comb_log(Y_true, Y_pred):
    loss_1 = loss_sq(Y_true, Y_pred)
    loss_2 = loss_pcc(Y_true, Y_pred)
    loss_3 = loss_log(Y_true, Y_pred)
    a1 = 50
    a2 = 50
    a3 = 1
    loss_value = (a1*loss_1+a2*loss_2+a3*loss_3)/(a1+a2+a3)
    return loss_value


# @tf.function
# def amp_constraint(amp):
#     import tensorflow as tf
#     amp = tf.where(amp > 1, tf.ones_like(amp), amp)
#     return amp


class amp_constraint(Layer):
    def call(self, inputs):
        out = tf.where(tf.logical_or(inputs>1,inputs<0), tf.ones_like(inputs), tf.zeros_like(inputs) )
        self.add_loss(tf.reduce_mean(out))
        return inputs

class phi_constraint(Layer):
    def call(self, inputs):
        import math
        out = tf.where(
            tf.math.abs(inputs) > math.pi, tf.ones_like(inputs),
            tf.zeros_like(inputs))
        self.add_loss(tf.reduce_mean(out))
        return inputs
