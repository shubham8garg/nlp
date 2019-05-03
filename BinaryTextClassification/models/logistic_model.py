#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression

def create_logistic_model(solver1='lbfgs'):
    classifier = LogisticRegression(solver=solver1)
    return classifier