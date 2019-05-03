#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_data, Y_data, solver1='lbfgs'):
    classifier = LogisticRegression(solver=solver1)
    classifier.fit(X_data, Y_data)
    return classifier