#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras import layers

def train_sequential_model(X_train, Y_train, X_test, Y_test):
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100, verbose=False, validation_data=(X_test, Y_test), batch_size=10)
    return model