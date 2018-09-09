import torch.utils.data as data  

import re
import requests
import numpy as np
from os import path
import string
import pickle

class TestDataSet(data.Dataset):
    def __init__(self, content = ''):
        self.word2id, self.id2word = get_vocabulary(content)        
        contexts, centers = training_data(content, self.word2id)

        self.contexts, self.centers = contexts, centers

    def __len__(self):
        return len(self.centers)
        
    def __getitem__(self, i):
        return np.array(self.contexts[i]), self.centers[i]
        
def get_vocabulary(content):
    '''
    Iterate over all articles, aggregating the vocabulary as we go
    '''
    s = set()
    word_list = content.split()
    for word in word_list:
        s.add(word)
    l = list(s)

    word2id = {w:i for i, w in enumerate(l)}
    id2word = {i:w for i, w in enumerate(l)}
    
    return word2id, id2word

def get_windows(words):
    '''
    The input words array 
    '''
    #center_id = int(win_size / 2)
    windows = []
    for k in range(1, len(words)-1):
        window = [words[k-1], words[k], words[k+1]]
        center = window[1]
        context = [window[0], window[2]]
        windows.append((context, center))
    return windows

def training_data(content, word2id):

    words = content.split()

    word_ids = [word2id[w] for w in words]
    windows = get_windows(word_ids)
            
    contexts = []
    centers = []

    for context, center in windows:
        contexts.append(context)
        centers.append(center)
    return contexts, centers
