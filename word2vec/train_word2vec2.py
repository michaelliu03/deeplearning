import jieba
import codecs
import word2vec
import logging

corpusFilePath = '../data/ming.all'
stopWordsFilePath = '../data/all_stopword.txt'
corpusHandlePath = '../data/ming_seg_filter.txt'
modelFilePath = '../model/'

fileTrain =[]
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def loaddata():
    with open(corpusFilePath,'r',encoding='utf-8') as f:
        for line in f:
            #print(line)
            fileTrain.append(line)
    f.close()

def get_stopWords(stopWords_fn):
    with open(stopWords_fn, 'r',encoding='utf-8') as f:
        stopWords_set = {line.strip('\r\t') for line in f}
        print(stopWords_set)
    return stopWords_set



def sentence2words(sentence):
    print(sentence)
    fileTrainSeg = []
    for i in range(len(sentence)):
        fileTrainSeg.append([' '.join(list(jieba.cut(sentence[i][9:-11], cut_all=False)))])


    with open(corpusHandlePath, 'w',encoding='utf-8') as fW:
        for i in fileTrainSeg:
           fW.write(' '.join(i))
           fW.write('\n')
           print(i)
           #fW.write(fileTrainSeg[i])
           #fW.write('\n')
    fW.close()

def trainmodel():

    word2vec.word2vec('../data/ming_seg_filter.txt','corpusWord2Vec.bin', size=100, verbose=True)




if __name__ == "__main__":
    trainmodel()
