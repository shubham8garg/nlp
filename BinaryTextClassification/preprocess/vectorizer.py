#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
vectorize_data:
    Input = n number of sentences
    Processing = It use the bag of words approach - CountVectorizer.
                Take all the words 'w' in 'n' sentences and create a dictionary of length 'w'. 
                Now for each sentence, it create vector of length 'w' and places the count of that word in place of the word index
    Output = n X w matrix 
    
    E.g. - Suppose you passed 5 utterances. Total number of words in all the sentences are 10. n = 5 and w = 10 
        "play" get a index of 2 and "music" gets a index of 7.
        A sentence "play play music" will get converted to - [0 0 2 0 0 0 0 1 0 0]
        Similarly, it will be a 5 X 10 matrix
'''
def vectorize_data(train, test):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(train)
    '''
    Can use:
        vectorizer.vocabulary_ : get the dictionary of mapping
        vectorizer.get_feature_names() : returns the list of words [] with length "w"
    '''
    return vectorizer, vectorizer.transform(train), vectorizer.transform(test)