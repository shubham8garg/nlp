#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras import layers

def create_emb_max_model(vocab_size, length, emb_dim=50):
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=length))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model