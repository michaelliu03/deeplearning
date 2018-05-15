import re
import os
import csv
import time
import codecs
import pickle
import numpy as np


class util(object):
    def __init__(self):
        self.char2id = None
        self.label2id = None
        self.id2char = None
        self.id2label = None

        self.inputIndex = None
        self.inputX = None
        self.inputY = None
        self.validX = None
        self.validY = None

    def  shuffle(self):
        self.inputIndex =0
        num_samples = len(self.inputX)
        indexs =np.arrange(num_samples)
        np.random.shuffle(indexs)




