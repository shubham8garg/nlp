import pandas as pd

def read1(filepath, sep1='\t'):
    total_data = pd.read_csv(filepath, names=['sentence', 'label'], sep=sep1)
    sentences = total_data['sentence'].values
    labels = total_data['label'].values
    return sentences, labels