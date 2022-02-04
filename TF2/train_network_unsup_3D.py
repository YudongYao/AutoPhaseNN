# seed random number generator
from numpy.random import seed

seed(123)
from tensorflow.random import set_seed

set_seed(123)

### Choose GPU settings, import libraries
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import numpy as np
import random
import time
import json
import datetime
import math

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K

from keras_helper import *
from forward_process import *
from data_generator import *
from clr_callback import CyclicLR

if __name__ == '__main__':

    t0 = time.time()
    # Training settings
    parser = argparse.ArgumentParser(
        description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # shared args
    # ============================================================
    parser.add_argument('--OutputFolder', type=str, default='./result/')
    # dataset
    parser.add_argument(
        '--DataFolder',
        type=str,
        default='/lus/theta-fs0/projects/Deep_WF/YYD/CDI_simulation_upsamp/')
    parser.add_argument('--num_workers',
                        default=4,
                        type=int,
                        help='num of workers')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--epoch', default=5, type=int, help='training epochs')
    parser.add_argument('--train_size',
                        type=int,
                        default=500,
                        help='training data size')
    parser.add_argument('--loss_type',
                        type=str,
                        default='mae',
                        help='loss type')
    parser.add_argument('--Initlr',
                        type=float,
                        default=1e-3,
                        help='initial lr')
    parser.add_argument('--lr_type', type=str, default='step', help='lr type')
    parser.add_argument('--T', type=float, default=0.1)
    parser.add_argument('--gpu_num', default=8, type=int, help='gpu num')
    parser.add_argument('--notes', type=str, default='')
    args = parser.parse_args()

    num_GPU = args.gpu_num
    print("Num GPUs Available: ",
          len(tf.config.experimental.list_physical_devices('GPU')))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if len(gpus):
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print("Restricting Memory")
        except RuntimeError as e:
            print(e)

    result_path = args.OutputFolder
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    datapath = args.DataFolder

    with open(os.path.join(args.OutputFolder, 'setting.json'), 'w') as f:
        f.write(json.dumps(args.__dict__, indent=4))

    for key, value in args.__dict__.items():
        print('{}: {}'.format(key, value))
    # print(args.__dict__)

    #### Some training parameters
    h, w, t = 64, 64, 64
    nepochs = args.epoch
    batch_size = args.batch_size

    mirrored_strategy = tf.distribute.MirroredStrategy()
    gpu_num = mirrored_strategy.num_replicas_in_sync
    batch_size = (batch_size * gpu_num)
    print('total batch_size is :%d' % batch_size)

    # # data generator
    dataname_list = os.path.join(datapath, '3D_upsamp.txt')
    filelist = []

    with open(dataname_list, 'r') as f:
        txtfile = f.readlines()
    for i in range(len(txtfile)):
        tmp = str(txtfile[i]).split('/')[-1]
        tmp = tmp.split('\n')[0]
        filelist.append(tmp)
    f.close()
    print('number of available file:%d' % len(filelist))

    # give training data size and filelist
    total_train_size = args.train_size
    train_filelist = filelist[:total_train_size]

    # Parameters
    params = {
        'dim': (64, 64, 64),
        'batch_size': batch_size,
        'n_classes': 1,
        'n_channels': 1,
        'shuffle': True,
        'load_all': False,
        'data_path': datapath,
        'data_name': ''
    }  #provide dataname for load_all=True

    # Datasets
    random.shuffle(train_filelist)
    print('training data size %d' % len(train_filelist))
    train_size = int(total_train_size * 0.9)
    validation_num = total_train_size - train_size

    train_id = train_filelist[:train_size]
    validation_id = train_filelist[train_size:]

    # Generators
    training_generator = DataGenerator(train_id, **params)
    validation_generator = DataGenerator(validation_id, **params)

    # SOME FUNCTIONS
    @tf.function
    def get_mask(input):
        import tensorflow as tf
        mask = tf.where(input >= args.T, tf.ones_like(input),
                        tf.zeros_like(input))
        return mask

    ## Define the network structure
    tf.keras.backend.clear_session()

    def get_compiled_model():

        fnum = 32

        input_img = Input(shape=(h, w, t, 1))

        # Encoding layers
        # Activations are all leakyReLu
        x = Conv_Pool_block(input_img,
                            fnum,
                            w1=3,
                            w2=3,
                            w3=3,
                            p1=2,
                            p2=2,
                            p3=2,
                            Lalpha=0.01,
                            padding='same',
                            data_format='channels_last')
        x = Conv_Pool_block(x,
                            fnum * 2,
                            w1=3,
                            w2=3,
                            w3=3,
                            p1=2,
                            p2=2,
                            p3=2,
                            Lalpha=0.01,
                            padding='same',
                            data_format='channels_last')
        x = Conv_Pool_block(x,
                            fnum * 4,
                            w1=3,
                            w2=3,
                            w3=3,
                            p1=2,
                            p2=2,
                            p3=2,
                            Lalpha=0.01,
                            padding='same',
                            data_format='channels_last')
        x = Conv_Pool_block(x,
                            fnum * 8,
                            w1=3,
                            w2=3,
                            w3=3,
                            p1=2,
                            p2=2,
                            p3=2,
                            Lalpha=0.01,
                            padding='same',
                            data_format='channels_last')
        x = Conv_Pool_block_last(x,
                                 fnum * 16,
                                 w1=3,
                                 w2=3,
                                 w3=3,
                                 p1=2,
                                 p2=2,
                                 p3=2,
                                 Lalpha=0.01,
                                 padding='same',
                                 data_format='channels_last')

        encoded = x

        #Decoding arm 1
        x1 = Conv_Upfirst_block(encoded,
                                fnum * 8,
                                w1=3,
                                w2=3,
                                w3=3,
                                p1=2,
                                p2=2,
                                p3=2,
                                Lalpha=0.01,
                                padding='same',
                                data_format='channels_last')
        x1 = Conv_Upfirst_block(x1,
                                fnum * 4,
                                w1=3,
                                w2=3,
                                w3=3,
                                p1=2,
                                p2=2,
                                p3=2,
                                Lalpha=0.01,
                                padding='same',
                                data_format='channels_last')
        x1 = Conv_Upfirst_block(x1,
                                fnum * 2,
                                w1=3,
                                w2=3,
                                w3=3,
                                p1=2,
                                p2=2,
                                p3=2,
                                Lalpha=0.01,
                                padding='same',
                                data_format='channels_last')
        x1 = Conv_Upfirst_block_last(x1,
                                     fnum * 1,
                                     w1=3,
                                     w2=3,
                                     w3=3,
                                     psize=16,
                                     padding='same',
                                     data_format='channels_last')

        decoded1 = Conv3D(1, (3, 3, 3), padding='same')(x1)
        decoded1 = Lambda(lambda x: sigmoid(x))(decoded1)

        # decoded1 = amp_constraint(name = 'amp')(decoded1)
        # decoded1 = Lambda(lambda x: amp_constraint(x), name='amp')(decoded1)
        support = Lambda(lambda x: get_mask(x), name='support')(decoded1)

        #Decoding arm 2
        x2 = Conv_Upfirst_block(encoded,
                                fnum * 4,
                                w1=3,
                                w2=3,
                                p1=2,
                                p2=2,
                                Lalpha=0.01,
                                padding='same',
                                data_format='channels_last')
        x2 = Conv_Upfirst_block(x2,
                                fnum * 4,
                                w1=3,
                                w2=3,
                                p1=2,
                                p2=2,
                                Lalpha=0.01,
                                padding='same',
                                data_format='channels_last')
        x2 = Conv_Upfirst_block(x2,
                                fnum * 2,
                                w1=3,
                                w2=3,
                                p1=2,
                                p2=2,
                                Lalpha=0.01,
                                padding='same',
                                data_format='channels_last')
        x2 = Conv_Upfirst_block_last(x2,
                                     fnum * 1,
                                     w1=3,
                                     w2=3,
                                     w3=3,
                                     psize=16,
                                     padding='same',
                                     data_format='channels_last')

        decoded2 = Conv3D(1, (3, 3, 3), padding='same')(x2)
        # decoded2 = phi_constraint(name='phi')(decoded2)
        decoded2 = Lambda(lambda x: math.pi * tanh(x), name='phi')(decoded2)

        # forward propagation
        obj = Lambda(lambda x: combine_complex(x[0], x[1]),
                     name='Obj')([decoded1, decoded2])
        masked_obj = Lambda(lambda x: x[0] * tf.cast(x[1], tf.complex64),
                            name='masked_obj')([obj, support])

        # far-field propagation to get the diff
        Psi = Lambda(lambda x: ff_propagation(x),
                     name='farfield_diff')(masked_obj)

        # Put together
        autoencoder = Model(input_img, Psi)

        opt = tf.keras.optimizers.Adam(learning_rate=args.Initlr)

        if args.loss_type == 'mae':
            autoencoder.compile(optimizer=opt, loss='mean_absolute_error')
        if args.loss_type == 'mae_cus':
            autoencoder.compile(optimizer=opt, loss=loss_mae)
        elif args.loss_type == 'mse':
            autoencoder.compile(optimizer=opt, loss=loss_sq)
        elif args.loss_type == 'huber':
            autoencoder.compile(optimizer=opt, loss=tf.keras.losses.Huber())
        elif args.loss_type == 'pcc':
            autoencoder.compile(optimizer=opt, loss=loss_pcc)
        elif args.loss_type == 'comb':
            autoencoder.compile(optimizer=opt, loss=loss_comb)
        elif args.loss_type == 'comb2':
            autoencoder.compile(optimizer=opt, loss=loss_comb2)
        else:
            autoencoder.compile(optimizer=opt, loss='mean_absolute_error')

        return autoencoder

    with mirrored_strategy.scope():
        autoencoder = get_compiled_model()

    # checkponts
    checkpoints = tf.keras.callbacks.ModelCheckpoint(
        '%s/weights.{epoch:02d}.hdf5' % result_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch')

    #Tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + os.path.basename(os.path.dirname(result_path))

    # tensorboard_callback = TensorBoard(log_dir=log_dir,histogram_freq=1)
    class LRTensorBoard(TensorBoard):
        def __init__(self, log_dir,
                     **kwargs):  # add other arguments to __init__ if you need
            super().__init__(log_dir=log_dir, **kwargs)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs.update({'lr': K.eval(self.model.optimizer.lr)})
            super().on_epoch_end(epoch, logs)

    # Learning rate
    if args.lr_type == 'Clr':
        EPOCHS = nepochs
        min_lr = 1e-7
        top_lr = args.Initlr
        step_size = int(train_size / batch_size) * 8
        gamma = 10**(np.log10(min_lr / top_lr) /
                     (train_size / batch_size * EPOCHS))
        reduce_lr = CyclicLR(base_lr=min_lr,
                             max_lr=top_lr,
                             step_size=step_size,
                             mode='exp_range',
                             gamma=gamma)
    elif args.lr_type == 'Step':

        def step_decay(epoch):
            initial_lrate = args.Initlr
            drop = 0.5
            epochs_drop = 20
            lrate = initial_lrate * math.pow(
                drop, math.floor((1 + epoch) / epochs_drop))
            # lrate = initial_lrate * np.exp(-epoch/epochs_drop)
            return lrate

        reduce_lr = tf.keras.callbacks.LearningRateScheduler(step_decay)

    elif args.lr_type == 'Plateau':
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.5,
                                                         patience=2,
                                                         min_lr=1e-7,
                                                         verbose=1)
    else:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.5,
                                                         patience=2,
                                                         min_lr=1e-7,
                                                         verbose=1)
    #start training
    history = autoencoder.fit(
        training_generator,
        validation_data=validation_generator,
        shuffle=True,
        batch_size=batch_size,
        use_multiprocessing=False,
        max_queue_size=128,
        workers=args.num_workers,
        verbose=2,
        epochs=nepochs,
        callbacks=[checkpoints, reduce_lr,
                   LRTensorBoard(log_dir=log_dir)])

    # save metrics of the training
    # epochs=np.asarray(history.epoch)+1
    # training_loss=history.history['loss']
    # val_loss=history.history['val_loss']
    np.save('%s/str_history' % result_path, history.history)

    #  Save the epoch number with the lowest validation loss
    val_losses = history.history['val_loss']
    min_epoch = np.argmin(val_losses) + 1
    print('minimum loss is epoch:%d' % min_epoch)
    np.save('%s/min_epoch' % result_path, min_epoch)

    # save learning rate
    # lr = reduce_lr.history['lr']
    # iteratin = reduce_lr.history['iterations']
    np.save('%s/str_lr' % result_path, reduce_lr.history)
