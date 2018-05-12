from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from sklearn.manifold import TSNE


fileName='../data/ming_seg_filter.txt'


def read_data(fileName):
    fileTrain =[]
    with open(fileName,'r',encoding='utf-8') as f:
        for line in f:
            print(line)
            fileTrain.append(line)
    f.close()
    return fileTrain




if __name__ =="__main__":
   read_data(fileName)
