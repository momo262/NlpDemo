import pandas as pd
import jieba
import numpy as np
import re
import array
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

def readtxtasarray(path):
    with open(path, 'r') as f:
        stopword = [line.strip() for line in f]
    return array(stopword)

#将句子列表进行分词，剔除停用词
def docs_to_wordlist(docs,stopwordlist, whitewordlist):
    #统计被剔除词
    deletedwordlist={}
    wordlist = []
    # 增加自定义词库
    jieba.load_userdict(readtxt('../data/自定义微博词库.txt'))

    # 统计表情词
    emojiwords = readtxt('../data/自定义微博词库.txt')
    for doc in docs:
        try:
            searchObj = re.search("\\[(.*?)\\]", doc)
        except TypeError as e :
            print("发生异常：" + str(doc))
            searchObj = None
        if (searchObj is not None):
            groups = searchObj.groups();
            if (groups.__len__() > 0):
                groups = searchObj.groups()
                for emoji in groups:
                    if emoji not in emojiwords :
                        emojiwords.add(emoji)

        words = []
        seg_list = jieba.cut(str(doc))
        for word in seg_list:

            if (word in whitewordlist) or (word not in stopwordlist and not word.isdigit()):
                words.append(word)
            else:
                #已经被剔除过，则+1，否则put进去
                if deletedwordlist.__contains__(word):
                    deletedwordlist[word] += 1
                else:
                    deletedwordlist[word] = 1
        wordlist.append(words)

    print(emojiwords)
    # 将收集到的词写入词库
    if (emojiwords.__len__() > 0) :
        f = open('../data/自定义微博词库.txt', 'a')
        for emoji in emojiwords:
        # f.writelines(emojiwords)
            if (emoji.startswith('[')) :
                f.write(emoji + "\n")
            else:
                f.write("[" + emoji + "]\n")
        f.close()
    deletedwordlist = sorted(deletedwordlist.items(), key=lambda x: x[1], reverse=True)
    print(deletedwordlist)
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


# inputs, outputs = readcsv('../data/ChnSentiCorp_htl_all.csv')
inputs, outputs = readcsv('../data/weibo_senti_100k.csv')
wordlist = docs_to_wordlist(inputs, readtxt('../data/中文停用词库.txt'), readtxt('../data/白名单词库.txt'))

count_out_put(outputs)
count_input_doc_length(wordlist)
