import numpy as np
import tensorflow.keras as keras
import scipy.io as sio

class DataGenerator(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, scenario, list_IDs, labels, batch_size=16, num_antennas=64,
                 num_subc=100, n_channels=2, shuffle=True, data_path="D:/code/mamimo_measurements_matlab/"):
        # 'Initialization'
        self.dim = (num_antennas, num_subc)
        self.antennas = [x for x in range(64)]

        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.data_path = data_path + scenario+"_channel_measurements" + "/"
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 2), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            sample = sio.loadmat(self.data_path + "channel_measurement_" + str(ID).zfill(6) + '.mat')['channel']
            # # print(X[i, :, :, 0].shape)
            # # print(sample.real.shape)
            X[i, :, :, 0] = sample.real[self.antennas, :]
            X[i, :, :, 1] = sample.imag[self.antennas, :]

            # Store class
            y[i,:] = self.labels[ID, :]

        return X, y
