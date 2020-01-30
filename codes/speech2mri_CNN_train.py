'''
Written by Tamas Gabor Csapo <csapot@tmit.bme.hu>
First version Jan 21, 2019

'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as io_wav
import os
import os.path
import datetime
import pickle
import cv2
import random

import vocoder_LSP_sptk

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Reshape

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# do not use all GPU memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True 
set_session(tf.Session(config=config))




# from LipReading with slight modifications
# https://github.com/hassanhub/LipReading/blob/master/codes/data_integration.py
################## VIDEO INPUT ##################
def load_video_3D(path, framesPerSec):
    
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # make sure that all the videos are the same FPS
    if (np.abs(fps - framesPerSec) > 0.01):
        print('fps:', fps, '(' + path + ')')
        raise

    buf = np.empty((frameHeight, frameWidth, frameCount), np.dtype('float32'))
    fc = 0
    ret = True
    
    while (fc < frameCount  and ret):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype('float32')
        # min-max scaling to [0-1]
        frame = frame-np.amin(frame)
        # make sure not to divide by zero
        if np.amax(frame) != 0:
            frame = frame/np.amax(frame)
        buf[:,:,fc]=frame
        fc += 1
    cap.release()

    return buf

# load vocoder features,
# or calculate, if they are not available
def get_mgc_lsp_coeff(basefilename):
    if os.path.isfile(basefilename + '.mgclsp'):
        mgc_lsp_coeff = np.fromfile(basefilename + '.mgclsp', dtype=np.float32).reshape(-1, order + 1)
        lf0 = np.fromfile(basefilename + '.lf0', dtype=np.float32)
    else:
        (mgc_lsp_coeff, lf0) = vocoder_LSP_sptk.encode(basefilename, samplingFrequency, frameLength, frameShift, order, alpha, stage)
    return (mgc_lsp_coeff, lf0)


for speaker in ['f1', 'f2', 'm1', 'm2']:
    # TODO: modify this according to your data path
    dir_mri = '/home/csapot/deep_learning_mri/usctimit_mri/' + speaker + '/'
    
    # Parameters of vocoder
    samplingFrequency = 20000
    frameLength = 1024 # 
    frameShift = 863 # 43.14 ms at 20000 Hz sampling, correspondong to 23.18 fps (MRI video)
    order = 24
    alpha = 0.42
    stage = 3
    n_mgc = order + 1

    # properties of MRI videos
    framesPerSec = 23.18
    n_width = 68
    n_height = 68
    

    # USC-TIMIT contains 92 files (460 sentences) for each speaker
    # train-valid-test split (random) :
    # - 4 files for valid
    # - 2 files for test
    # - the remaining (86 files) for training
    files_mri = dict()
    mri = dict()
    mgc = dict()
    files_mri['all'] = []
    if os.path.isdir(dir_mri):
        for file in sorted(os.listdir(dir_mri)):
            if ".avi" in file:
                files_mri['all'] += [file]
    
    # randomize file order
    random.seed(17)
    random.shuffle(files_mri['all'])
    
    files_mri['valid'] = files_mri['all'][0:4]
    files_mri['test'] = files_mri['all'][4:6]
    files_mri['train'] = files_mri['all'][6:]
    
    print('valid files', files_mri['valid'])
    print('test files', files_mri['test'])   # ['usctimit_mri_f1_146_150.avi', 'usctimit_mri_f1_441_445.avi']
    
    for train_valid in ['train', 'valid']:
        n_files = len(files_mri[train_valid])
        n_file = 0
        n_max_mri_frames = n_files * 1000
        mri[train_valid] = np.empty((n_max_mri_frames, n_width, n_height))
        mgc[train_valid] = np.empty((n_max_mri_frames, n_mgc))
        mri_size = 0
        mgc_size = 0

        for file in files_mri[train_valid]:
            try:
                print('starting', train_valid, file)
                mri_data = load_video_3D(dir_mri + file, framesPerSec)
                (mgc_lsp_coeff, lf0) = get_mgc_lsp_coeff(dir_mri + file[:-4])
            except ValueError as e:
                print("wrong data, check manually!", e)
            
            else:
                print('minmax:', np.min(mri_data), np.max(mri_data))
                n_file += 1
                
                mgc_mri_len = np.min((mri_data.shape[2], len(mgc_lsp_coeff)))
                
                mri_data = mri_data[:, :, 0:mgc_mri_len]
                mgc_lsp_coeff = mgc_lsp_coeff[0:mgc_mri_len]
                
                if mri_size + mgc_mri_len > n_max_mri_frames:
                    raise
                
                for i in range(mgc_mri_len):
                    mri[train_valid][mri_size + i] = mri_data[:, :, i] # original, 68x68
                    mgc[train_valid][mgc_size + i] = mgc_lsp_coeff[i]
                
                mri_size += mgc_mri_len
                mgc_size += mgc_mri_len
                
                print('n_frames_all: ', mri_size, 'mgc_size: ', mgc_size)
                    
        mri[train_valid] = mri[train_valid][0 : mri_size].reshape(-1, n_width, n_height, 1)
        mgc[train_valid] = mgc[train_valid][0 : mgc_size]


    
    # target: min max scaler to [0,1] range
    # already scaled in load_video
    
    # input: normalization to zero mean, unit variance
    # feature by feature
    mgc_scalers = []
    for i in range(n_mgc):
        mgc_scaler = StandardScaler(with_mean=True, with_std=True)
        mgc_scalers.append(mgc_scaler)
        mgc['train'][:, i] = mgc_scalers[i].fit_transform(mgc['train'][:, i].reshape(-1, 1)).ravel()
        mgc['valid'][:, i] = mgc_scalers[i].transform(mgc['valid'][:, i].reshape(-1, 1)).ravel()


    ### single training
    model = Sequential()
    
    model.add(Dense(500, input_dim=n_mgc, kernel_initializer='normal', activation='relu'))
    model.add(Dense(17*17*8, kernel_initializer='normal', activation='relu'))
    model.add(Reshape((17, 17, 8)))
    
    model.add(UpSampling2D(size=2))
    model.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
    
    model.add(UpSampling2D(size=2))
    model.add(Conv2D(1, kernel_size=3, padding='same', activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    print(model.summary())
    
    current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() )
    model_name = 'models/SPEECH2MRI_CNN_' + speaker + '_' + current_date

    print('starting training', speaker, current_date)

    # early stopping to avoid over-training
    # csv logging of loss
    # save best model
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0), \
                 CSVLogger(model_name + '.csv', append=True, separator=';'),
                 ModelCheckpoint(model_name + '_weights.h5', monitor='val_loss')]

    # run training
    history = model.fit(mgc['train'], mri['train'],
                            epochs = 100, batch_size = 128, shuffle = True, verbose = 1,
                            validation_data=(mgc['valid'], mri['valid']),
                            callbacks=callbacks)

    # here the training of the DNN is finished


    # save model
    model_json = model.to_json()
    with open(model_name + '_model.json', "w") as json_file:
        json_file.write(model_json)

    # serialize scalers to pickle
    pickle.dump(mgc_scalers, open(model_name + '_mgc_scalers.sav', 'wb'))
    
    # save test files
    with open(model_name + '_test_files.txt', 'w') as txt_file:
        for file in files_mri['test']:
            txt_file.write(file + '\n')
    
    print('finished training', speaker, current_date)
    