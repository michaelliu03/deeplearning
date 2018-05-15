import re
import os
import csv
import time
import codecs
import pickle
import numpy as np


class Util(object):
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
        self.inputX = self.inputX[indexs]
        self.inputY = self.inputY[indexs]

    def initFile(self, inputPath, validPath=None, seqMaxLen=200):
        self.inputIndex = 0
        self.inputX, self.inputY = self.loadFile(inputPath, seqMaxLen)

        # shuffle the samples
        # self.shuffle()

        if validPath != None:
            self.validFile = open(validPath)
            self.validX, self.validY = self.loadFile(validPath, seqMaxLen)

        return (self.inputX, self.inputY) if validPath == None else (self.inputX, self.inputY, self.validX, self.validY)

    def loadFile(self, inputPath, seqMaxLen=200):
        X = []
        Y = []
        x = []
        y = []
        for line in open(inputPath,'r',encoding='utf-8'):
            line = line.strip()
            if len(line) == 0:
                if len(x) <= seqMaxLen and len(x) > 0:
                    X.append(x)
                    Y.append(y)
                x = []
                y = []
                continue

            terms = line.split()
            char = terms[0]
            label = terms[1]
            c = self.char2id["<NEW>"] if char not in self.char2id else self.char2id[char]
            l = -1 if label not in self.label2id else self.label2id[label]
            x.append(c)
            y.append(l)

        X = np.array(self.padding(X, seqMaxLen))
        Y = np.array(self.padding(Y, seqMaxLen))
        return X, Y



if __name__ == "__main__":
    util = Util()
    util.loadFile( '../data/ChinaLangOrg.seg.label',200)
    #print("kai x")





