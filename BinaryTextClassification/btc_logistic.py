#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

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

def read_file(filepath):
    total_data = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    sentences = total_data['sentence'].values
    labels = total_data['label'].values
    return sentences, labels    

def vectorize_data(train, test):
    vectorizer = CountVectorizer()
    vectorizer.fit(train)
    return vectorizer.transform(train), vectorizer.transform(test)

def train_logistic_regression(X_train, Y_train, X_test, Y_test):
    classifier = LogisticRegression(solver='lbfgs')
    classifier.fit(X_train, Y_train)
    print("Logistic Regression Accuracy score: {}".format(classifier.score(X_test, Y_test)))

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
    train_logistic_regression(X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    main()