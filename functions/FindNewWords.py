from gensim.models import fasttext
from gensim.models import word2vec
import pandas as pd
import logging
import jieba
import functions.ReadDataFile as rdf
import numpy as np
import gensim as gs

print()

# inputs, outputs = rdf.readcsv('../data/weibo_senti_100k.csv')
# wordlist = rdf.docs_to_wordlist(inputs, rdf.readtxt('../data/中文停用词库.txt'))
# model = word2vec.Word2Vec(wordlist, size=300, window=5, min_count=3, workers=4,sg=1)
# model.save("../data/word2vecLarge_skipgram.model")
# # model.wv.save_word2vec_format('./mymodel.txt', binary=False)
# print('打印与空间最相近的5个词语：', model.most_similar('空间', topn=5))

# model = gs.models.Word2Vec.load("../data/word2vec.model")
# vocab_list = [word for word, Vocab in model.wv.vocab.items()]# 存储 所有的 词语
