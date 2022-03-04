import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
num_GPU = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.models import Model, load_model

import math
import sys
import glob

# disable GUI backend
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import random

from scipy.ndimage.measurements import center_of_mass as com
from scipy.ndimage.interpolation import shift
from skimage.restoration import unwrap_phase

from keras_helper import *
from forward_process import *
from data_generator import *

print(os.environ["CUDA_VISIBLE_DEVICES"])
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if len(gpus):
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("Restricting Memory")
    except RuntimeError as e:
        print(e)


@tf.function
def get_mask(input):
    import tensorflow as tf

    mask = tf.where(input >= 0.1, tf.ones_like(input), tf.zeros_like(input))
    return mask


@tf.function
def loss_comb2(Y_true, Y_pred):
    loss_1 = tf.math.sqrt(loss_sq(Y_true, Y_pred))
    loss_2 = loss_pcc(Y_true, Y_pred)
    a1 = 1
    a2 = 1
    loss_value = (a1 * loss_1 + a2 * loss_2) / (a1 + a2)
    return loss_value


# load model
result_name = './results/trained_model_simulation.hdf5'  # the saved model path
model = load_model(result_name,
                   custom_objects={
                       'loss_comb': loss_comb,
                       'loss_comb2': loss_comb2,
                       'loss_sq': loss_sq,
                       'amp_constraint': amp_constraint,
                       'phi_constraint': phi_constraint
                   })

# load test data
datapath = './CDI_simulation_upsamp_noise/'
dataname_list = datapath + '3D_upsamp.txt'

filelist = []
with open(dataname_list, 'r') as f:
    txtfile = f.readlines()
for i in range(len(txtfile)):
    tmp = str(txtfile[i]).split('/')[-1]
    tmp = tmp.split('\n')[0]

    filelist.append(tmp.split('.npy')[0])
f.close()
print(len(filelist))

batch_size = 16
test_size = len(filelist) - 50000

print(test_size)
test_filelist = filelist[-test_size:]
random.shuffle(test_filelist)

# Parameters
params = {
    'dim': (64, 64, 64),
    'batch_size': batch_size,
    'n_classes': 1,
    'n_channels': 1,
    'shuffle': False,
    'load_all': False,
    'load_test': True,
    'data_path': datapath,
    'data_name': ''
}

test_id = test_filelist

test_generator = DataGenerator(test_id, **params)
test_generator.set_batch(batch_size)
preds_intens = model.predict(test_generator, verbose=1)

# # predicted object
obj_layer_model = Model(inputs=model.input,
                        outputs=model.get_layer('masked_obj').output)
preds_obj = obj_layer_model.predict(test_generator, verbose=1)
preds_amp = np.abs(preds_obj)
ph_layer_model = Model(inputs=model.input,
                       outputs=model.get_layer('phi').output)
preds_phi = ph_layer_model.predict(test_generator, verbose=1)
support_layer_model = Model(inputs=model.input,
                            outputs=model.get_layer('support').output)
preds_support = support_layer_model.predict(test_generator, verbose=1)

h, w, t = 64, 64, 64


def shift_com(amp, phi):
    coms = com(amp)
    if (np.isnan(coms[0]) or np.isnan(coms[1])):
        return shape, 0
    deltas = (int(round(h / 2 - coms[0])), int(round(w / 2 - coms[1])),
              int(round(t / 2 - coms[2])))
    amp_shift = shift(amp, shift=deltas, mode='wrap')
    phi_shift = shift(phi, shift=deltas, mode='wrap')
    return amp_shift, phi_shift


def shift_sup(support):
    coms = com(support)
    if (np.isnan(coms[0]) or np.isnan(coms[1])):
        return shape, 0
    deltas = (int(round(h / 2 - coms[0])), int(round(w / 2 - coms[1])),
              int(round(t / 2 - coms[2])))
    support = shift(support, shift=deltas, mode='wrap')
    return support


def post_process(amp, phi, th=0.1, uw=0, flag=0):
    amp = amp.reshape(h, w, t)
    phi = phi.reshape(h, w, t)

    if uw == 1:
        #         phi = np.unwrap(np.unwrap(np.unwrap(phi,0),1),2)
        phi = unwrap_phase(phi)

    mask = np.where(amp > th, 1, 0)
    amp_out = mask * amp
    phi_out = mask * phi

    mean_phi = np.mean(phi_out[amp_out > th])
    phi_out = phi_out - mean_phi
    amp_out, phi_out = shift_com(amp_out, phi_out)

    mask = np.where(amp_out > th, 1, 0)
    amp_out = mask * amp_out
    phi_out = mask * phi_out

    if flag == 1:
        phi_out = np.where(mask == 0, np.nan, phi_out)

    return amp_out, phi_out


# save 10 randomly chosen test examples
h, w, t = 64, 64, 64
index = 32
ntest = preds_intens.shape[0]
# plt.viridis()
n = 10
f, ax = plt.subplots(10, n, figsize=(22, 20))

for i in range(0, n):
    j = int(round(np.random.rand() * ntest))

    test_generator.set_batch(1)
    [X_test, [Y_I_test, Y_phi_test]] = test_generator.__getitem__(j)

    Y_I_test, Y_phi_test = post_process(Y_I_test,
                                        Y_phi_test,
                                        th=0.1,
                                        uw=1,
                                        flag=0)

    Y_I_pred, Y_phi_pred = post_process(preds_amp[j],
                                        preds_phi[j],
                                        th=0.1,
                                        uw=1,
                                        flag=0)

    support = preds_support[j].reshape(h, w, t)
    support = shift_sup(support)

    mask = np.where(Y_I_pred[:, :, index].reshape(h, w) > 0.1, 1, 0)

    # display input FT intensity
    im = ax[0, i].imshow(np.log10(X_test[0, :, :, index].reshape(h, w) + 1))
    plt.colorbar(im, ax=ax[0, i], format='%.2f')
    ax[0, i].get_xaxis().set_visible(False)
    ax[0, i].get_yaxis().set_visible(False)

    # display predicted FT intensity
    im = ax[1,
            i].imshow(np.log10(preds_intens[j, :, :, index].reshape(h, w) + 1))
    plt.colorbar(im, ax=ax[1, i], format='%.2f')
    ax[1, i].get_xaxis().set_visible(False)
    ax[1, i].get_yaxis().set_visible(False)

    #Difference in FT intensity
    im = ax[2, i].imshow(X_test[0, :, :, index].reshape(h, w) -
                         preds_intens[j, :, :, index].reshape(h, w))
    plt.colorbar(im, ax=ax[2, i], format='%.2f')
    ax[2, i].get_xaxis().set_visible(False)
    ax[2, i].get_yaxis().set_visible(False)

    # display original amplitude
    im = ax[3, i].imshow(Y_I_test[:, :, index].reshape(h, w))
    plt.colorbar(im, ax=ax[3, i], format='%.2f')
    ax[3, i].get_xaxis().set_visible(False)
    ax[3, i].get_yaxis().set_visible(False)

    # display predicted amplitude
    im = ax[4, i].imshow(Y_I_pred[:, :, index].reshape(h, w))
    plt.colorbar(im, ax=ax[4, i], format='%.2f')
    ax[4, i].get_xaxis().set_visible(False)
    ax[4, i].get_yaxis().set_visible(False)

    # Difference in amplitude
    im = ax[5, i].imshow(Y_I_test[:, :, index].reshape(h, w) -
                         Y_I_pred[:, :, index].reshape(h, w))
    plt.colorbar(im, ax=ax[5, i], format='%.2f')
    ax[5, i].get_xaxis().set_visible(False)
    ax[5, i].get_yaxis().set_visible(False)

    # display original phase
    im = ax[6, i].imshow(Y_phi_test[:, :, index].reshape(h, w))
    plt.colorbar(im, ax=ax[6, i], format='%.2f')
    ax[6, i].get_xaxis().set_visible(False)
    ax[6, i].get_yaxis().set_visible(False)

    # display predicted phase
    im = ax[7, i].imshow(mask * (Y_phi_pred[:, :, index]).reshape(h, w))
    plt.colorbar(im, ax=ax[7, i], format='%.2f')
    ax[7, i].get_xaxis().set_visible(False)
    ax[7, i].get_yaxis().set_visible(False)

    # Difference in phase
    im = ax[8, i].imshow(Y_phi_test[:, :, index].reshape(h, w) - mask *
                         (Y_phi_pred[:, :, index].reshape(h, w)))
    plt.colorbar(im, ax=ax[8, i], format='%.2f')
    ax[8, i].get_xaxis().set_visible(False)
    ax[8, i].get_yaxis().set_visible(False)

    im = ax[9, i].imshow(support[:, :, index].reshape(h, w))
    plt.colorbar(im, ax=ax[9, i], format='%.2f')
    ax[9, i].get_xaxis().set_visible(False)
    ax[9, i].get_yaxis().set_visible(False)

f.savefig(wt_path + '/result_example.png', dpi=600)
