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
from preprocess.vectorizer import vectorize_data
from models.logistic_model import create_logistic_model

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
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    #Read the whole file
    X, Y = read1(args.file)

    #Splitting into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_size, random_state=1000)

    #Vectorize the data
    vectorizer, X_train, X_test = vectorize_data(X_train, X_test)

    #Create a model
    classifier = create_logistic_model()

    #Train a model
    classifier.fit(X_train, Y_train)
    
    #Print the scores
    print("Logistic Regression Train accuracy: {}".format(classifier.score(X_train, Y_train)))
    print("Logistic Regression Test accuracy: {}".format(classifier.score(X_test, Y_test)))

if __name__ == "__main__":
    main()