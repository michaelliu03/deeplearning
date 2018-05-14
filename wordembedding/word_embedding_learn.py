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


fileName='../data/tmall_vip_ZCM_corpus_seg.txt'


def read_data(fileName):
    fileTrain =[]
    with open(fileName,'r',encoding='utf-8') as f:
        for line in f:
            #print(line)
            fileTrain.append(line)
    f.close()
    return fileTrain

vocabulary_size= 50000
def build_dataset(words):
    count =[['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary =dict()
    for word,_ in count:
        #print(word)
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(read_data(fileName))

data_index =0
def generate_batch(batch_size, num_skips, skip_window):
      global data_index
      assert batch_size % num_skips == 0
      assert num_skips <= 2 * skip_window
      batch = np.ndarray(shape=(batch_size), dtype=np.int32)
      labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
      span = 2 * skip_window + 1 # [ skip_window target skip_window ]
      buffer = collections.deque(maxlen=span)
      for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
      for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
          while target in targets_to_avoid:
            target = random.randint(0, span - 1)
          targets_to_avoid.append(target)
          batch[i * num_skips + j] = buffer[skip_window]
          labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
      return batch, labels



#if __name__ =="__main__":
   #read_data(fileName)
   #build_dataset(read_data(fileName))
   # print('data:', [reverse_dictionary[di] for di in data[:8]])
   #
   # for num_skips, skip_window in [(2, 1), (4, 2)]:
   #     data_index = 0
   #     batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
   #     print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
   #     print('    batch:', [reverse_dictionary[bi] for bi in batch])
   #     print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

