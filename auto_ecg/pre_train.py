import sys, os
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint,
                             TensorBoard, ReduceLROnPlateau,
                             CSVLogger, EarlyStopping)
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
sys.path.append(os.path.abspath("../models"))

from model import model
import argparse
from keras.utils import HDF5Matrix
import pandas as pd
import h5py
import numpy as np

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
    print (tf.test.is_gpu_available())
    
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
#     parser.add_argument('path_to_hdf5', type=str,
#                         help='path to hdf5 file containing tracings')
#     parser.add_argument('path_to_csv', type=str,
#                         help='path to csv file containing annotations')
#################################
    parser.add_argument('path_train_x', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_train_y', type=str,
                        help='path to csv file containing annotations')
    parser.add_argument('path_save_model', type=str, default = "./final_model.hdf5",
                        help='save model file name')    
#################################
    parser.add_argument('--val_split', type=float, default=0.02,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. The remaining '
                             'is used for validation. Default: 0.02')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    args = parser.parse_args()
    # Optimization settings
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]
    # Set session and compile model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    # If you are continuing an interrupted section, uncomment line bellow:
    ### pretrain model 
    model_6labels = load_model('./model.hdf5', compile=False)
    weight = model_6labels.get_weights()

    weight[-2] = np.hstack([weight[-2], np.random.rand(weight[-2].shape[0], 3)])
    bias = np.random.rand(9)
    bias[:6] = weight[-1]
    weight[-1] = bias

    model.set_weights(weight)
    model.compile(loss=loss, optimizer=opt)
    # Get annotations
#     y = pd.read_csv(args.path_to_csv).values
#     # Get tracings
#     f = h5py.File(args.path_to_hdf5, "r")
#     x = f[args.dataset_name]
    y = np.load(args.path_train_y)
    x = np.load(args.path_train_x)


    # Create log
    callbacks += [TensorBoard(log_dir='./logs', batch_size=batch_size, write_graph=False),
                  CSVLogger('training.log', append=False)]  # Change append to true if continuing training
    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./backup_model_last.hdf5'),
                  ModelCheckpoint('./backup_model_best.hdf5', save_best_only=True)]
    # Train neural network
    
    for layer in model.layers:
        layer.trainable = False

    model.layers[-1].trainable = True
    model.layers[-2].trainable = True
    
    history = model.fit(x, y,
                        batch_size=batch_size,
                        epochs=35,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        validation_split=args.val_split,
                        shuffle='batch',  # Because our dataset is an HDF5 file
                        callbacks=callbacks,
                        verbose=1)
    
    for layer in model.layers:
        layer.trainable = True
        
    history = model.fit(x, y,
                        batch_size=batch_size,
                        epochs=35,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        validation_split=args.val_split,
                        shuffle='batch',  # Because our dataset is an HDF5 file
                        callbacks=callbacks,
                        verbose=1)
    
    # Save final result
    model.save(args.path_save_model)
#     f.close()
