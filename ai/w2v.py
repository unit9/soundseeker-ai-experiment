import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import numpy as np
from gensim.models import KeyedVectors


def get_words(row, tag_columns):
    words = []
    for tag in tag_columns:
        if (row[tag] > 0).bool():
            words.append((row[tag].values[0], tag[4:]))
    words.sort(reverse=True)
    for i, word in enumerate(words):
        w = word[1]
        if "_" in w:
            w = w.split("_")[0]
        words[i] = w
    return words


def load_w2v_model(model_path="w2v_slim.bin.gz"):
    print("start loading w2v model")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("finish loading w2v model")
    return model


def get_words_vector(words, model, num=2):
    tmp = []
    for word in words:
        try:
            tmp.append(model[word])
        except Exception:
            print("word {} not in set".format(word))
        if len(tmp) >= num:
            break
    if not len(tmp):
        tmp.append(model['nothing'])
    return np.mean(tmp, axis=0)
