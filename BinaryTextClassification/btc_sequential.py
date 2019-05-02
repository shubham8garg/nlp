#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt


def create_parser():
    description = 'Add some description'
    example_use = 'Example: script.py -f <file>'
    parser = argparse.ArgumentParser(description=description, epilog=example_use)
    # =================================== #
    # Required arguments for main command #
    # =================================== #
    required_options_group = parser.add_argument_group(title='Required arguments for main command')
    required_options_group.add_argument('--file', '-f', required=True, help='File to be passed to this script')
    # ================================= #
    # Optional options for main command #
    # ================================= #
    optional_options_group = parser.add_argument_group(title='Optional arguments for main command')
    optional_options_group.add_argument('--test_size', '-t', required=False, help='File to be passed to this script', default=0.25, type=float)
    optional_options_group.add_argument('--plot', '-p', required=False, help='File to be passed to this script', default=False, action='store_true')
    return parser

def read_file(filepath):
    total_data = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    sentences = total_data['sentence'].values
    labels = total_data['label'].values
    return sentences, labels    

def vectorize_data(train, test):
    vectorizer = CountVectorizer()
    vectorizer.fit(train)
    return vectorizer.transform(train), vectorizer.transform(test)

def train_sequential_model(X_train, Y_train, X_test, Y_test):
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=100, verbose=False, validation_data=(X_test, Y_test), batch_size=10)
    train_loss, train_accuracy = model.evaluate(X_train, Y_train, verbose=False)
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=False)
    print("Sequential Model Training Accuracy: {}".format(train_accuracy))
    print("Sequential Model Test Accuracy: {}".format(test_accuracy))
    return history

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('seq_model.png')
    

def main():
    parser = create_parser()
    args = parser.parse_args()

    #Read the whole file
    X, Y = read_file(args.file)

    #Splitting into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_size, random_state=1000)

    #Vectorize the data
    X_train, X_test = vectorize_data(X_train, X_test)

    #Train a model
    history = train_sequential_model(X_train, Y_train, X_test, Y_test)

    #Plot
    if args.plot == True:
        plt.style.use('ggplot')
        plot_history(history)

if __name__ == "__main__":
    main()