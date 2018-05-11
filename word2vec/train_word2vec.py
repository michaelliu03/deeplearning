import sys
import codecs
from gensim.models import Word2Vec
import jieba
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_stopWords(stopWords_fn):
    with open(stopWords_fn, 'r',encoding='utf-8') as f:
        stopWords_set = {line.strip('\r\t') for line in f}
        print(stopWords_set)
    return stopWords_set


def sentence2words(sentence, stopWords=False, stopWords_set=None):
    """
    split a sentence into words based on jieba
    """
    # seg_words is a generator
    seg_words = jieba.cut(sentence)
    if stopWords:
        words = [word for word in seg_words if word not in stopWords_set and word != ' ']
    else:
        words = [word for word in seg_words]
    return words


class MySentences(object):
    def __init__(self, list_csv):
        stopWords_fn = 'all_stopword.txt'
        self.stopWords_set = get_stopWords(stopWords_fn)
        all_ = codecs.open(list_csv,'r',encoding='utf-8')
        self.fns = [line.strip() for line in all_]

    def __iter__(self):
        for fn in self.fns:
            with open(fn, 'r',encoding='utf-8') as f:
                for line in f:
                    yield sentence2words(line.strip(), True, self.stopWords_set)


def train_save(list_csv, model_fn):
    stopWords_fn = 'all_stopword.txt'
    stopWords_set = get_stopWords(stopWords_fn)
    all_ = codecs.open(list_csv, 'r', encoding='utf-8')
    fns = [line.strip() for line in all_]


    sentences = fns
    num_features = 100
    min_word_count = 10
    num_workers = 48
    context = 20
    epoch = 20
    sample = 1e-5
    model = Word2Vec(
        sentences,
        size=num_features,
        min_count=min_word_count,
        workers=num_workers,
        sample=sample,
        window=context,
        iter=epoch,
    )
    model.save(model_fn)
    return model


if __name__ == "__main__":
    #stopWords_fn = 'all_stopword.txt'
    #get_stopWords(stopWords_fn)
    #model = train_save('ming.all', 'word2vec_model_0925')

    model = Word2Vec.load('word2vec_model_0925')

    #model_1 = Word2Vec.load(model)
    # 计算两个词的相似度/相关程度
    y1 = model.similarity("春", "夏")
    print(u"春和夏的相似度为：", y1)
    print("-------------------------------\n")

    y2 =  model.wv['春']
    print(y2)
