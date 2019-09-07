import pandas as pd
import jieba
import numpy as np

fo = open("../formatData/weiboSentiment_train.txt", "w")
fo2 = open("../formatData/weiboSentiment_text.txt", "w")
fo3 = open("../formatData/weiboSentiment_eval.txt", "w")

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

def write_file(inputs,outputs):
    rate = np.array([0.8, 0.1, 0.1])
    cumsum_rate = np.cumsum(rate)
    index = 0

    for wordlist in inputs:
        random = int(np.searchsorted(cumsum_rate, np.random.rand(1) * 1.0))
        sentence = ""
        for word in wordlist:
            sentence = sentence + word + " "
        label = outputs[index]
        index = index + 1

        if random == 0:
            fo.write(sentence.strip() + '_label_')
            fo.write(str(label))
            fo.write("\n")

        if random == 1:
            fo2.write(sentence.strip() + '_label_')
            fo2.write(str(label))
            fo2.write("\n")

        if random == 2:
            fo3.write(sentence.strip() + '_label_')
            fo3.write(str(label))
            fo3.write("\n")

# def find_new_words():


inputs, outputs = readcsv('../data/weibo_senti_100k.csv')
wordlist = docs_to_wordlist(inputs, readtxt('../data/中文停用词库.txt'))
write_file(wordlist, outputs)
count_out_put(outputs)
count_input_doc_length(wordlist)
