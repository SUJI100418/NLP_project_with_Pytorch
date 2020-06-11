"""
서로 다른 강연 스크립트 3개.
그 중 강연별 중요 단어 추출하기
"""

import pandas as pd
from operator import itemgetter
import numpy as np
from doc import doc1,doc2,doc3

# 문서 내 단어들의 출현 빈도를 세는 함수.기 (문서 1개)
def get_term_frequency(document, word_dict=None):
    if word_dict is None:
        word_dict = {}
    words = document.split()

    for w in words:
        word_dict[w] = 1+ (0 if word_dict.get(w) is None else word_dict[w]) # 해당 단어 갯수 얻

    return pd.Series(word_dict).sort_values(ascending=False)

# 각 단어가 몇개의 문서에서 나타났는지 세는 함수. (여러 문서)
def get_document_frequency(documents):
    dicts = []
    vocab = set([])
    df = {}

    for d in documents:
        tf = get_term_frequency(d)
        dicts += [tf]
        vocab = vocab | set (tf.keys())

    for v in list(vocab):
        df[v] = 0
        for dict_d in dicts:
            if dict_d.get(v) is not None:
                df[v] += 1

    return pd.Series(df).sort_values(ascending=False)

def get_tfidf(docs):
    vocab = {}
    tfs = []
    for d in docs:
        vocab = get_term_frequency(d, vocab)
        tfs += [get_term_frequency(d)]
    df = get_document_frequency(docs)

    stats = []
    for word, freq in vocab.items():
        tfidfs = []
        for idx in range(len(docs)):
            if tfs[idx].get(word) is not None:
                tfidfs += [tfs[idx][word] * np.log(len(docs) / df[word])]
            else:
                tfidfs += [0]
        stats.append((word, freq, *tfidfs, max(tfidfs)))

    return pd.DataFrame(stats, columns=('word', # 단어.
                                        'frequency', # 총 출현 횟수.
                                        'doc1', # tf-idf 문서1
                                        'doc2',
                                        'doc3',
                                        'max')).sort_values('max',ascending=False)

#print(get_tfidf([doc1,doc2,doc3]))  # 모든 단어 별, 문서별 중요 단어 추출.


# tf 값(단어의 문서별 출현 횟수)을 특징 벡터로 사용하기.
# 어떤 단어가 문서마다 출현한 횟수가 차원별로 구성

def get_tf(docs):
    vocab = {}
    tfs = []
    for d in docs:
        vocab = get_term_frequency(d, vocab)
        tfs += [get_term_frequency(d)]

    stats = []
    for word, freq in vocab.items():
        tf_v = []
        for idx in range(len(docs)):
            if tfs[idx].get(word) is not None:
                tf_v += [tfs[idx][word]]
            else:
                tf_v += [0]
        stats.append((word, freq, *tf_v))

    return pd.DataFrame(stats, columns=('word',
                                       'frequency',
                                       'doc1',
                                       'doc2',
                                       'doc3')).sort_values('frequency', ascending=False)

print(get_tf([doc1,doc2,doc3]))