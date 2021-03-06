#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras import layers

def create_sequential_model(input_dim):
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model