#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.preprocessing.sequence import pad_sequences

def pad_data(data, maxlen=100, padding='post'):
    new_data = pad_sequences(data, padding=padding, maxlen=maxlen)
    return new_data