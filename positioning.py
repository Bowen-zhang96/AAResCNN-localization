import tensorflow as tf

import numpy as np
from tensorflow.keras import initializers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, Conv2D, Concatenate, Add, Multiply, Reshape, Dropout, Conv3D, LeakyReLU,AveragePooling2D, ReLU,Conv2DTranspose, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler

import scipy.io as sio
from tensorflow.keras import optimizers
from attn_augconv import AttentionAugmentation2D
# import keras
# import pickle as p
from data_generator import DataGenerator

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

transfered = "ULA"
scenarios = ["ULA"] #, "URA"
num_antennas = [64]
# scenario = "URA"
data_path = "C:/code_new/mamimo_measurements_matlab/"

# tf.logging.set_verbosity(tf.logging.ERROR)

# Distance Functions
def dist(y_true, y_pred):
    return tf.reduce_mean((
        tf.sqrt(
            tf.square(tf.abs(y_pred[:, 0] - y_true[:, 0]))
            + tf.square(tf.abs(y_pred[:, 1] - y_true[:, 1]))
        )))


def true_dist(y_true, y_pred):
    return np.sqrt(
        np.square(np.abs(y_pred[:, 0] - y_true[:, 0]))
        + np.square(np.abs(y_pred[:, 1] - y_true[:, 1]))
    )



# Definition of the NN
def build_nn(num_antenna=64):
    nn_input = Input((num_antenna, num_sub, 2))


    layD1 = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(nn_input)
    layD1 = LeakyReLU(alpha=0.3)(layD1)

    # RB1
    layD1_ = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1)
    layD1_ = LeakyReLU(alpha=0.3)(layD1_)
    layD1_ = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1_)
    layD1 = Add()([layD1, layD1_])
    # PB1
    layD1 = Conv2D(64, (5, 5), strides=(1, 1), padding='same')(layD1)
    layD1 = AveragePooling2D((2, 2), strides=(2, 2), padding='same')(layD1)
    layD1 = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1)

    # RB2
    layD1_ = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1)
    layD1_ = LeakyReLU(alpha=0.3)(layD1_)
    layD1_ = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1_)
    layD1 = Add()([layD1, layD1_])
    # PB2
    layD1 = Conv2D(64, (5, 5), strides=(1, 1), padding='same')(layD1)
    layD1 = AveragePooling2D((2, 2), strides=(2, 2), padding='same')(layD1)
    layD1 = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1)

    # layD1_long_connection=layD1
    # AARB1
    conv_out = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1)
    qkv_conv = Conv2D(4 * 3, (1, 1), strides=(1, 1), padding='same')(layD1)
    attn_out = AttentionAugmentation2D(4, 4, 1, relative=True)(qkv_conv)
    attn_out = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding='same')(attn_out)
    layD1_ = Concatenate(axis=-1)([conv_out, attn_out])
    layD1_ = LeakyReLU(alpha=0.3)(layD1_)
    layD1_ = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1_)
    layD1 = Add()([layD1, layD1_])

    # AARB2
    conv_out = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1)
    qkv_conv = Conv2D(4 * 3, (1, 1), strides=(1, 1), padding='same')(layD1)
    attn_out = AttentionAugmentation2D(4, 4, 1, relative=True)(qkv_conv)
    attn_out = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding='same')(attn_out)
    layD1_ = Concatenate(axis=-1)([conv_out, attn_out])
    layD1_ = LeakyReLU(alpha=0.3)(layD1_)
    layD1_ = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1_)
    layD1 = Add()([layD1, layD1_])

    # AARB3
    conv_out = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1)
    qkv_conv = Conv2D(4 * 3, (1, 1), strides=(1, 1), padding='same')(layD1)
    attn_out = AttentionAugmentation2D(4, 4, 1, relative=True)(qkv_conv)
    attn_out = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding='same')(attn_out)
    layD1_ = Concatenate(axis=-1)([conv_out, attn_out])
    layD1_ = LeakyReLU(alpha=0.3)(layD1_)
    layD1_ = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1_)
    layD1 = Add()([layD1, layD1_])

    # AARB4
    conv_out = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1)
    qkv_conv = Conv2D(4 * 3, (1, 1), strides=(1, 1), padding='same')(layD1)
    attn_out = AttentionAugmentation2D(4, 4, 1, relative=True)(qkv_conv)
    attn_out = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding='same')(attn_out)
    layD1_ = Concatenate(axis=-1)([conv_out, attn_out])
    layD1_ = LeakyReLU(alpha=0.3)(layD1_)
    layD1_ = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1_)
    layD1 = Add()([layD1, layD1_])

    # AARB5
    conv_out = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1)
    qkv_conv = Conv2D(4 * 3, (1, 1), strides=(1, 1), padding='same')(layD1)
    attn_out = AttentionAugmentation2D(4, 4, 1, relative=True)(qkv_conv)
    attn_out = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding='same')(attn_out)
    layD1_ = Concatenate(axis=-1)([conv_out, attn_out])
    layD1_ = LeakyReLU(alpha=0.3)(layD1_)
    layD1_ = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1_)
    layD1 = Add()([layD1, layD1_])

    # AARB6
    conv_out = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1)
    qkv_conv = Conv2D(4 * 3, (1, 1), strides=(1, 1), padding='same')(layD1)
    attn_out = AttentionAugmentation2D(4, 4, 1, relative=True)(qkv_conv)
    attn_out = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding='same')(attn_out)
    layD1_ = Concatenate(axis=-1)([conv_out, attn_out])
    layD1_ = LeakyReLU(alpha=0.3)(layD1_)
    layD1_ = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1_)
    layD1 = Add()([layD1, layD1_])

    # AARB7
    conv_out = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1)
    qkv_conv = Conv2D(4 * 3, (1, 1), strides=(1, 1), padding='same')(layD1)
    attn_out = AttentionAugmentation2D(4, 4, 1, relative=True)(qkv_conv)
    attn_out = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding='same')(attn_out)
    layD1_ = Concatenate(axis=-1)([conv_out, attn_out])
    layD1_ = LeakyReLU(alpha=0.3)(layD1_)
    layD1_ = Conv2D(32, (5, 5), strides=(1, 1), padding='same')(layD1_)
    layD1 = Add()([layD1, layD1_])

    #



    nn_output = Flatten()(layD1)
    nn_output = Dense(64, activation='linear')(nn_output)
    nn_output = LeakyReLU(alpha=0.3)(nn_output)
    nn_output = Dense(32, activation='linear')(nn_output)
    nn_output = LeakyReLU(alpha=0.3)(nn_output)
    nn_output = Dense(2, activation='linear')(nn_output)
    nn = Model(inputs=nn_input, outputs=nn_output)
    adam=optimizers.Adam(learning_rate=1e-3,clipvalue=10.)
    nn.compile(optimizer=adam, loss='mse', metrics=[dist])
    nn.summary()
    return nn


num_samples = 252004


validation_size = 0.5                     # 50% for validation
test_size = 0.5                          # 50% for testing

num_sub = 100

labels = sio.loadmat(data_path + 'labels.mat')['lables']
mean_x=np.mean(labels[:,0])
mean_y=np.mean(labels[:,1])
labels[:,0]=labels[:,0]-mean_x
labels[:,1]=labels[:,1]-mean_y


for scenario in scenarios:
    # check for bad channels (channels with corrupt data)
    bad_samples = sio.loadmat("C:/code_new/mamimo_measurements_matlab/bad_channels_" + scenario + ".mat")['bad_channels']
    # buils array with all valid channel indices

    # Generate the train/test dataset with random sampling
    # IDs_all = []
    # for x in range(num_samples):
    #     if x not in bad_samples:
    #         IDs_all.append(x)
    # IDs_all = np.array(IDs_all)
    # # shuffle the indices with fixed seed
    # np.random.seed(64)
    # np.random.shuffle(IDs_all)
    # IDs=IDs[:2000]

    IDs = sio.loadmat('ULA_test.mat')['ID']  # 2000
    IDs = IDs[0]


    actual_num_samples = IDs.shape[0]

    val_IDs = IDs[:int(validation_size * actual_num_samples)]
    test_IDs = IDs[-int(test_size * actual_num_samples):]

    train_IDs=sio.loadmat('ID_1000_ULA_train.mat')['ID']
    train_IDs=train_IDs[0]

    for num_antenna in num_antennas:
        print("scenario:", scenario, "number of antennas:", num_antenna)
        nn = build_nn(num_antenna)

        # I used batch size 128 for all experiments, but when training data size is smaller. a smaller batch size should
        # be used
        batch_size = 64

        val_generator = DataGenerator(scenario, val_IDs, labels,shuffle=False,
                                      num_antennas=num_antenna,
                                      batch_size=batch_size,
                                      data_path=data_path)
        test_generator = DataGenerator(scenario, test_IDs, labels,
                                        shuffle=False, num_antennas=num_antenna, batch_size=batch_size,
                                        data_path=data_path)
        nb_epoch = 1000


        val_dist_hist = []
        train_dist_hist = []
        # # #
        # try:
        #     nn = load_model('bestmodels/best_model_ifft_' + scenario + '_' + str(num_antenna) + '_10000_final.h5',
        #                     custom_objects={"tf": tf, "dist": dist,"AttentionAugmentation2D":AttentionAugmentation2D})
        # except Exception:
        #     print("Couldn't load weights")

        # # # simple early stopping

        mc = ModelCheckpoint('bestmodels/best_model_ifft_' + scenario + '_' + str(num_antenna) + '_1000.h5', monitor='val_dist', mode='min', verbose=1, save_best_only=True)

        # I decayed learning rate every 200 epochs in original paper,
        # but I found the results could be better if changing 200 to 400 after paper submission
        def schedule(epoch):
            if epoch%400==0 and epoch>0:
                lr=K.get_value(nn.optimizer.lr)
                if lr>0.000125:
                    K.set_value(nn.optimizer.lr,lr*0.5)
                    print('lr changed to {}'.format(lr*0.5))
            return K.get_value(nn.optimizer.lr)
        reduce_lr=LearningRateScheduler(schedule)

        train_generator = DataGenerator(scenario, train_IDs, labels,
                                        batch_size=batch_size,
                                        num_antennas=num_antenna,
                                        data_path=data_path)
        train_hist = nn.fit_generator(train_generator, epochs=nb_epoch,
                                      validation_data=val_generator,
                                      callbacks=[reduce_lr,mc],initial_epoch=0,validation_freq=10)

        val_dist_hist.extend(train_hist.history['val_dist'])
        train_dist_hist.extend(train_hist.history['dist'])
        np.save('positioning_model_ifft_' + scenario + '_' + str(num_antenna) + '_1000.npy', nn.get_weights())
        np.save('val_dist_hist_ifft_' + scenario + '_' + str(num_antenna) + '_1000.npy', val_dist_hist)
        np.save('train_dist_hist_ifft_' + scenario + '_' + str(num_antenna) + '_1000.npy', train_dist_hist)
        # plot training history

        # # Load best model to evaluate it's performance on the test set
        nn = load_model('bestmodels/best_model_ifft_' + scenario + '_' + str(num_antenna) + '_1000.h5', custom_objects={"tf": tf, "dist": dist,"AttentionAugmentation2D":AttentionAugmentation2D})

        r_Positions_pred_test = nn.predict_generator(test_generator)
        test_length = r_Positions_pred_test.shape[0]
        test_IDs_integer=np.zeros([np.shape(test_IDs)[0]],dtype=np.int32)
        for i in range(np.shape(test_IDs)[0]):
            test_IDs_integer[i]=int(test_IDs[i])
        tmp=test_IDs_integer[:test_length]
        r_Positions_true_test=labels[tmp]
        errors_test = true_dist(r_Positions_true_test, r_Positions_pred_test)
        Mean_Error_Test = np.mean(np.abs(errors_test))
        print('\033[1m{:<40}{:.4f}\033[0m'.format('Performance P: Mean error on Test area: ', Mean_Error_Test), 'mm')
        # sio.savemat('test_dataset_URA_10000.mat',{'pred_test':r_Positions_pred_test,'true_test':r_Positions_true_test})

