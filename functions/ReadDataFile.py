import pandas as pd
import jieba
import numpy as np

#读取csv文件
def readcsv(path):
    pd_all = pd.read_csv(path)
    inputs = []
    outputs = []
    for index in pd_all.index:
        inputs.append(pd_all.loc[index].review)
        outputs.append(pd_all.loc[index].label)
    return inputs, outputs

#读取txt文件
def readtxt(path):
    with open(path, 'r') as f:
        stopword = [line.strip() for line in f]
    return set(stopword)

#将句子列表进行分词，剔除停用词
def docs_to_wordlist(docs,stopwordlist):
    wordlist = []
    for doc in docs:
        words = []
        seg_list = jieba.cut(str(doc))
        for word in seg_list:
            if word not in stopwordlist and not word.isdigit():
                words.append(word)
        wordlist.append(words)
    return wordlist

#统计标签的类别分布
def count_out_put(outputs):
    print('评论数目（总体）：%d' % len(outputs))
    print('评论数目（正向）：%d' % outputs.count(1))
    print('评论数目（负向）：%d' % outputs.count(0))

#统计句子长度分布
def count_input_doc_length(wordlist):
    lengths = []
    maxlength = 0
    for words in wordlist:
        length = 0
        for word in words:
            length = length + 1
        lengths.append(length)
        if length > maxlength:
            maxlength = length
    print('最长句子词数：%d' % maxlength)
    print('平均词数及标准差')
    print(np.mean(lengths), np.std(lengths))

# def find_new_words():


inputs, outputs = readcsv('../data/ChnSentiCorp_htl_all.csv')
wordlist = docs_to_wordlist(inputs, readtxt('../data/中文停用词库.txt'))
count_out_put(outputs)
count_input_doc_length(wordlist)
