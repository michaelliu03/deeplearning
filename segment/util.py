import re
import os
import csv
import time
import codecs
import pickle
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
        logging.INFO("initFile starting.......")
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

    def bulidMap(self,trainPath):
        charSet = set()
        labelSet = set()
        for line  in open(trainPath,'r', encoding='utf-8'):
            line = line.strip()
            if len(line)  == 0 : continue

            terms = line.split()
            if len(terms)  != 2:  continue
            charSet.add(terms[0]) #char
            labelSet.add(terms[1]) #label

        chars = list(charSet)
        labels = list(labelSet)

        self.char2id = dict(zip(chars,range(1,len(chars) +1)))
        self.label2id = dict(zip(labels,range(1,len(labels) + 1)))

        self.id2char = dict(zip(range(1,len(chars) + 1),chars))
        self.id2label = dict(zip(range(1,len(labels) + 1),labels))

        self.id2char[0]  = "<PAD>"
        self.id2label[0] = "<PAD>"
        self.char2id["<PAD>"]  = 0
        self.label2id["<PAD>"] = 0
        self.id2char[len(chars) + 1] = "<NEW>"
        self.char2id["<NEW>"] = len(chars) + 1

        # save map
        with open("char2id", "wb",encoding='utf-8') as outfile:
            for idx in self.id2char:
                outfile.write(self.id2char[idx] + "\t" + str(idx) + "\r\n")
        with open("label2id", "wb",encoding='utf-8') as outfile:
            for idx in self.id2label:
                outfile.write(self.id2label[idx] + "\t" + str(idx) + "\r\n")
        print( "saved map between token and id")

    def loadMap(self, filePath):
        if not os.path.isfile(filePath):
            sys.exit("map file not exist")



if __name__ == "__main__":
    util = Util()
    util.bulidMap( '../data/seg/ChinaLangOrg.seg.label')
    #print("kai x")





