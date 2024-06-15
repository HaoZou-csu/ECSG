# -*- coding: utf-8 -*-

'''
@Time    : 2021/9/17 15:45
@Author  : Zou Hao

'''
import os
import numpy as np
import pandas as pd
from feature_engineering import Elfrac_fea, Magpie_fea, Meredig_fea, ElemNet_fea, ATCNN_fea, ECCNN_fea
from xgboost.sklearn import XGBClassifier
import keras
from keras import Sequential
from keras.layers import Dense,Activation,BatchNormalization,Dropout,Conv2D,MaxPooling2D,Flatten

def meta_fea_1(formulas):
    filepath = '../model/Elfrac_'
    pre_y = []
    X_1 = Elfrac_fea(formulas)
    for i in range(10):
        modelpath = filepath + str(i)
        model = XGBClassifier(n_estimators=200,max_depth=5)
        model.load_model(modelpath)
        y = model.predict_proba(X_1)
        pre_y.append(y)
    return pre_y

def meta_fea_2(formulas):
    filepath = '../model/Magpie_'
    pre_y = []
    X_1 = Elfrac_fea(formulas)
    for i in range(10):
        modelpath = filepath + str(i)
        model = XGBClassifier(n_estimators=200,max_depth=5)
        model.load_model(modelpath)
        y = model.predict_proba(X_1)
        pre_y.append(y)
    return pre_y

def meta_fea_3(formulas):
    filepath = '../model/Meredig_'
    pre_y = []
    X_1 = Elfrac_fea(formulas)
    for i in range(10):
        modelpath = filepath + str(i)
        model = XGBClassifier(n_estimators=200,max_depth=5)
        model.load_model(modelpath)
        y = model.predict_proba(X_1)
        pre_y.append(y)
    return pre_y

def meta_fea_4(formulas):
    filepath = '../model/ElemNet_'
    pre_y = []
    X_4 = ElemNet_fea(formulas)
    for i in range(10):
        modelpath = filepath + str(i) + '.best.hdf5'
        model = Sequential()
        model.add(Dense(1024, input_dim=86))
        model.add(Activation('relu'))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(1,activation='sigmoid'))

        model = keras.models.load_model(modelpath)
        y = model.predict(X_4)
        pre_y.append(y)
    return pre_y

def meta_fea_4(formulas):
    filepath = '../model/ElemNet_'
    pre_y = []
    X_4 = ElemNet_fea(formulas)
    for i in range(10):
        modelpath = filepath + str(i) + '.best.hdf5'
        model = Sequential()
        model.add(Dense(1024, input_dim=86))
        model.add(Activation('relu'))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(1,activation='sigmoid'))

        model = keras.models.load_model(modelpath)
        y = model.predict(X_4)
        pre_y.append(y)
    return pre_y

def meta_fea_6(formulas):
    filepath = '../model/ATCNN_'
    pre_y = []
    X_6 = ATCNN_fea(formulas)
    input_shape = (10, 10, 1)

    for i in range(10):
        modelpath = filepath + str(i) + '.best.hdf5'
        model = Sequential()

        # layer1
        model.add(Conv2D(64, kernel_size=(5, 5), padding='same', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # layer2
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # layer3
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # layer4
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # layer5
        model.add(Conv2D(64, kernel_size=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # layer6
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(200))

        # layer7
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Activation('relu'))

        model.add(Dense(100))

        # layer8
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Activation('relu'))

        # layer9
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model = keras.models.load_model(modelpath)
        y = model.predict(X_6)
        pre_y.append(y)
    return pre_y

def meta_fea_7(formulas):
    filepath = '../model/ECCNN_'
    pre_y = []
    X_7 = ECCNN_fea(formulas)
    for i in range(10):
        modelpath = filepath + str(i) + '.best.hdf5'
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', data_format='channels_first', activation='relu',
                         input_shape=(8, 118, 168)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        # model.add(Dense(2048,activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        # model.add(Dense(128,activation='relu'))
        # model.add(Dense(32,activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model = keras.models.load_model(modelpath)
        y = model.predict(X_7)
        pre_y.append(y)
    return pre_y

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    df = pd.read_csv('../data/wolverton_oxides.csv')
    formulas = df['formula'].values

    pre_y = meta_fea_7(formulas)
    print(pre_y)