'''
Written by Tamas Gabor Csapo <csapot@tmit.bme.hu>
First version Jan 21, 2019
Restructured Jan 21, 2020 - for MRI data

Keras implementation of Csapó T.G., ,,Speaker dependent acoustic-to-articulatory inversion using real-time MRI of the vocal tract'', accepted at Interspeech 2020

code for inference (MRI video generation)
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as io_wav
import os
import os.path
import glob
import pickle

import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc

from subprocess import call, check_output, run

import vocoder_LSP_sptk

from keras.models import model_from_json


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

# convert an array of values into a dataset matrix
# code with modifications from
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def create_dataset_img(data_in_X, data_in_Y, look_back=1):
    (dim1_X, dim2_X, dim3_X, dim4_X) = data_in_X.shape
    (dim1_Y, dim2_Y) = data_in_Y.shape
    data_out_X = np.empty((dim1_X - look_back - 1, look_back, dim2_X, dim3_X, dim4_X))
    data_out_Y = np.empty((dim1_Y - look_back - 1, dim2_Y))
    
    for i in range(dim1_X - look_back - 1):
        for j in range(look_back):
            data_out_X[i, j] = data_in_X[i + j]
        data_out_Y[i] = data_in_Y[i + j]
    return data_out_X, data_out_Y

# convert an array of values into a dataset matrix
# code with modifications from
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def create_dataset_img_inverse(data_in_X, data_in_Y, look_back=1):
    (dim1_X, dim2_X) = data_in_X.shape
    (dim1_Y, dim2_Y, dim3_Y, dim4_Y) = data_in_Y.shape
    data_out_X = np.empty((dim1_X - look_back - 1, look_back, dim2_X))
    data_out_Y = np.empty((dim1_Y - look_back - 1, dim2_Y, dim3_Y, dim4_Y))
    
    for i in range(dim1_X - look_back - 1):
        for j in range(look_back):
            data_out_X[i, j] = data_in_X[i + j]
        data_out_Y[i] = data_in_Y[i + j]
    return data_out_X, data_out_Y

# mri2vid converts raw MRI data to .mp4 video
def mri2vid(mri_data, dir_file, filename_no_ext, n_width, n_height, FramesPerSec):
    
    print(filename_no_ext + ' - MRI video started')
    
    output_file_no_ext = dir_file + filename_no_ext
    n_frames = len(mri_data)
    
    # compressed
    # fourcc = VideoWriter_fourcc(*'MP4V')
    
    # uncompressed 8-bit
    fourcc = VideoWriter_fourcc(*'Y800')
    video = VideoWriter(output_file_no_ext + '.avi', fourcc, float(FramesPerSec), (n_width, n_height), 0)
    
    for n in range(n_frames):
        frame = np.uint8(255 * mri_data[n]).reshape(n_width, n_height, 1)
        
        video.write(frame)
        print('frame ', n, ' done', end='\r')
    
    video.release()
    
    print(filename_no_ext + ' - MRI video finished')

def mrividwav2demo(dir_mri, file_mri, dir_wav, file_wav): 
    # "-codec copy " + \
    command = "ffmpeg " + \
           "-y " + \
           "-i " + dir_mri + file_mri + " " + \
           "-i " + dir_wav + file_wav + " " + \
           "-shortest " + \
           "-acodec copy -vcodec copy " + \
            dir_mri + file_mri[:-4] + "_with_audio.avi"
           # "-c:v h264 -crf 20 -c:a aac -strict -2 " + \
           # "-filter:v \"crop=820:496:215:48\" " + \
    
    print(command)
    run(command, shell=True)

# for speaker in ['f1']: # ['f1', 'f2', 'm1', 'm2']:
for speaker in ['f1', 'f2', 'm1', 'm2']:
    # TODO: modify this according to your data path
    dir_mri = '/home/csapot/deep_learning_mri/usctimit_mri/' + speaker + '/'
    dir_mri_test = 'generated_image_sequence/' + speaker + '/'
    
    if not os.path.exists(dir_mri_test):
        os.makedirs(dir_mri_test)
    
    # Parameters of vocoder
    samplingFrequency = 20000
    frameLength = 1024 # 
    frameShift = 863 # 43.14 ms at 20000 Hz sampling, correspondong to 23.18 fps (MRI video)
    order = 24
    alpha = 0.42
    stage = 3
    n_mgc = order + 1

    # context window of LSTM
    n_sequence = 10

    # properties of MRI videos
    framesPerSec = 23.18
    n_width = 68
    n_height = 68
    
    DNN_types = ['FC-DNN_baseline', 'CNN', 'LSTM']
    # DNN_types = ['FC-DNN_baseline', 'CNN']
    # DNN_types = ['LSTM-CNN']
    basefilenames_mri_test = ['usctimit_mri_' + speaker + '_146_150', 'usctimit_mri_' + speaker + '_441_445']
    
    for DNN_type in DNN_types:
        # e.g. MRI2SPEECH_CNN_f1_2020-01-16_10-36-35
        csv_files = glob.glob('models/SPEECH2MRI_' + DNN_type + '_' + speaker + '*.csv')
        model_name = csv_files[-1][:-4]
        
        # load model
        print('loading model', model_name)
        with open(model_name + '_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(model_name + '_weights.h5')
        # load scalers
        mgc_scalers = pickle.load(open(model_name + '_mgc_scalers.sav', 'rb'))
        
        for basefilename in basefilenames_mri_test:
            print('testing on: ', basefilename)
            
            # load data for sentence
            mri_data = load_video_3D(dir_mri + basefilename + '.avi', framesPerSec)
            mri_len = mri_data.shape[2]
            mri_test = np.empty((mri_len, n_width, n_height))
            (mgc_lsp_coeff, lf0) = get_mgc_lsp_coeff(dir_mri + basefilename)
            
            for i in range(mri_len):
                mri_test[i] = mri_data[:, :, i] # original, 68x68
            
            # transform of input parameters
            for i in range(n_mgc):
                mgc_lsp_coeff[:, i] = mgc_scalers[i].transform(mgc_lsp_coeff[:, i].reshape(-1, 1)).ravel()
            
            # reshape for LSTM
            if DNN_type == 'LSTM' or DNN_type == 'LSTM-CNN':
                mgc_len = len(mgc_lsp_coeff)
                mri0 = np.empty((mgc_len, n_width, n_height, 1))
                mgc_test0, mri0 = create_dataset_img_inverse(mgc_lsp_coeff, mri0, look_back = n_sequence)
                mgc_test = np.empty((mgc_len, n_sequence, n_mgc))
                
                # add first n_sequence values
                for i in range(mgc_len - 2):
                    if i < n_sequence - 0:
                        mgc_test[i] = mgc_test0[0]
                    else:
                        mgc_test[i] = mgc_test0[i - n_sequence + 1]
                
                mgc_lsp_coeff = mgc_test
            
            # predict MR image sequence using the trained model
            mri_predicted = model.predict(mgc_lsp_coeff)
            
            # clip extreme values
            mri_predicted = np.clip(mri_predicted, 0, 1)
            
            print(mri_predicted.shape)
            
            
            # save image sequence to video (without audio)
            mri2vid(mri_predicted, dir_mri_test, basefilename + '_' + DNN_type, n_width, n_height, framesPerSec)
            
            # put together video and audio
            mrividwav2demo(dir_mri_test, basefilename + '_' + DNN_type + '.avi', \
                dir_mri, basefilename + '.wav')