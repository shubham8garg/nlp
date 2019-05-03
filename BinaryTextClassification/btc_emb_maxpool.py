#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####
# Importing libraries
####
import argparse
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

####
#Importing functions from other scripts
####
from read_write_file.read_two_columns import read1
from preprocess.tokenizer import tokenize_data
from preprocess.padder import pad_data
from models.emb_maxpool import create_emb_max_model

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

def main():
    parser = create_parser()
    args = parser.parse_args()

    #Read the whole file
    X, Y = read1(args.file)

    #Splitting into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_size, random_state=1000)

    #Tokenize the data
    tokenizer, X_train, X_test = tokenize_data(X_train, X_test, max_words=5000)

    #Pad the sequences
    maxlen = 100
    X_train = pad_data(X_train, maxlen)
    X_test = pad_data(X_test, maxlen)

    #Create a model
    model = create_emb_max_model(vocab_size=(len(tokenizer.word_index) + 1), length=maxlen)

    #Train a model                                                   
    model_history = model.fit(X_train, Y_train, epochs=100, verbose=False, validation_data=(X_test, Y_test), batch_size=10)   
    
    #Calculate Accuracy and Print that
    train_loss, train_accuracy = model.evaluate(X_train, Y_train, verbose=False)
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=False)
    print("Sequential Model Training Accuracy: {}".format(train_accuracy))
    print("Sequential Model Test Accuracy: {}".format(test_accuracy))

    #Plot
    if args.plot == True:
        from plots.plot_model import plot_history
        plot_history(model_history)

if __name__ == "__main__":
    main()