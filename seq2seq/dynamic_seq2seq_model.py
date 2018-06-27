# -*- coding:utf-8 -*-
import math
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMStateTuple


class dynamicSeq2seq():
    '''
    Dynamic_Rnn_Seq2seq with Tensorflow-1.0.0

       args:
       encoder_cell            encoder结构
       decoder_cell            decoder结构
       encoder_vocab_size      encoder词典大小
       decoder_vocab_size      decoder词典大小
       embedding_size          embedd成的维度
       bidirectional           encoder的结构
                               True:  encoder为双向LSTM
                               False: encoder为一般LSTM
       attention               decoder的结构
                               True:  使用attention模型
                               False: 一般seq2seq模型
       time_major              控制输入数据格式
                               True:  [time_steps, batch_size]
                               False: [batch_size, time_steps]
    '''
    PAD = 0
    EOS = 2
    UNK = 3

    def __init__(self, encoder_cell,
                 decoder_cell,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 embedding_size,
                 bidirectional=True,
                 attention=False,
                 debug=False,
                 time_major=False):
        self.debug = debug
        self.bidirectional = bidirectional
        self.attention = attention
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size

        self.embedding_size = embedding_size
        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self.global_step = tf.Variable(-1,trainable=False)
        self.max_gradient_norm = 5
        self.time_major = time_major

        # 创建模型
        self._make_graph()

