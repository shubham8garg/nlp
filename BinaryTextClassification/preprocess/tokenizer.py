#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
tokenize_data:
    Input = 'n' number of sentences, 'num' number of words to consider.
    Processing = Take all the words 'w' in 'n' sentences and create a dictionary of length 'w' with unique word_index for each word.
                 Then select only top 'num' from 'w' dictionary and assign another index to all remaining words. 
                 Now for each sentence, it create vector of sentence length and places word index of that word in the place of the word.
    Output = n X varied length (based on the sentence).
    
    E.g. - Suppose you passed 5 utterances. Total number of words in all the sentences are 10. n = 5 and w = 10 
        "play" get a index of 2 and "music" gets a index of 7.
        A sentence "play play music" will get converted to - [2 2 7]

    Important: 0 is a reserved index and will not be assigned
                Tokenizer automatically has filters such as punctuations to remove from words. !"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n
                It only use num - 1 words and use the last one for remaining words
'''
def tokenize_data(train, test, num):
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=num)
    tokenizer.fit_on_texts(train)

    new_train = tokenizer.texts_to_sequences(train)
    new_test = tokenizer.texts_to_sequences(test)
    
    '''
    Can use:
        tokenizer.word_index : get the dictionary of mapping
        vocab_size = len(tokenizer.word_index) + 1   # As it reserve the 0 index
    '''
    return tokenizer, new_train, new_test
