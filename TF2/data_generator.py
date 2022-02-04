import tensorflow.keras.utils as keras_u
import numpy as np
import tensorflow as tf

class DataGenerator(keras_u.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=1, dim=(64,64,64), out_dim=(64,64,64), n_channels=1,
                 n_classes=1, shuffle=True, load_all=False,load_test=False,data_path='',data_name=''):
        'Initialization'
        self.dim = dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.load_all = load_all
        self.datapath = data_path
        self.test = load_test


        if self.load_all==True:
            
            data_all_path = data_path
            self.diff = np.load(self.datapath+data_name)[self.list_IDs[0]:self.list_IDs[-1]+1]
                
            print((self.diff).shape)
            
            if self.test == True:
                self.Y_I = np.load(self.datapath+'Y_I_test.npy')[self.list_IDs[0]:self.list_IDs[-1]+1]
                self.Y_phi = np.load(self.datapath+'Y_phi_test.npy')[self.list_IDs[0]:self.list_IDs[-1]+1]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def set_batch(self,batch_size):
        self.batch_size=batch_size
        
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        if self.load_all == True:
            list_IDs_temp = [k for k in indexes]
        else:
            list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        if self.test == True:
            X,Y = self.__data_generation_test(list_IDs_temp)
        else:
            X,Y = self.__data_generation(list_IDs_temp)

        return X,Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        Y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)

        if self.load_all == True:
            for i, ID in enumerate(list_IDs_temp):
                X[i,:,:,:,0]=self.data[ID,:,:,:,0]
                Y[i,:,:,:,0]=self.data[ID,:,:,:,0]
        else:
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                diff = np.load(self.datapath+ID)['arr_0']
                X[i,:,:,:,0] = diff
                Y[i,:,:,:,0] = diff

              
        return X,Y
    
    def __data_generation_test(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        Y_I = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        Y_phi = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)

        if self.load_all == True:
            for i, ID in enumerate(list_IDs_temp):
                X[i,:,:,:,0]=self.diff[ID,:,:,:,0]
                Y_I[i,:,:,:,0]=self.Y_I[ID,:,:,:,0]
                Y_phi[i,:,:,:,0]=self.Y_phi[ID,:,:,:,0]
        else:
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                diff = np.load(self.datapath+ID)['arr_0']
                data = np.load(self.datapath+ID)['arr_1']
                X[i,:,:,:,0] = diff
                Y_I[i,:,:,:,0] = np.abs(data)
                Y_phi[i,:,:,:,0] = np.angle(data)

              
        return X,[Y_I,Y_phi]