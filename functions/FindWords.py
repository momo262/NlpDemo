import re
import functions.ReadDataFile as rdf
import pandas as pd

def readtxt(path):
    with open(path, 'r') as f:
        stopword = [line.strip() for line in f]
    return set(stopword)

def find_emoji_words(docs):
    # 统计表情词
    emojiwords= set()
    # emojiwords = readtxt('../data/自定义微博词库.txt')
    newemojiwords = set()
    # 检测
    for doc in docs:
        try:
            searchObj = re.search("\\[(.*?)\\]", doc)
        except TypeError as e:
            print("发生异常：" + str(doc))
            searchObj = None
        if (searchObj is not None):
            groups = searchObj.groups();
            if (groups.__len__() > 0):
                print("\n原句：" + str(doc))
                groups = searchObj.groups()
                for emoji in groups:
                    if (emoji not in emojiwords) & ("[" + emoji + "]" not in emojiwords) & (emoji.__len__() < 10):
                        print("emoji word:" + emoji)
                        print("add emoji")
                        emojiwords.add(emoji)
                        newemojiwords.add(emoji)
    print(newemojiwords)
    # 将收集到的词写入词库
    # if (emojiwords.__len__() > 0):
    file = open('../data/自定义微博词库.txt', 'a')
    for emoji in newemojiwords:
        # f.writelines(emojiwords)
        if (emoji.startswith('[')):
            file.write(emoji + "\n")
        else:
            file.write("[" + emoji + "]\n")
    file.close()


def find_topic_words(docs):
    # 统计表情词
    # emojiwords={}
    emojiwords = readtxt('../data/微博话题词库.txt')
    newemojiwords = set()
    # 检测
    for doc in docs:
        try:
            searchObj = re.search("\\#(.*?)\\#", doc)
        except TypeError as e:
            print("发生异常：" + str(doc))
            searchObj = None
        if (searchObj is not None):
            groups = searchObj.groups();
            if (groups.__len__() > 0):
                print("\n原句：" + str(doc))
                groups = searchObj.groups()
                for emoji in groups:
                    if (emoji not in emojiwords) & ("#" + emoji + "#" not in emojiwords) & (emoji.__len__() < 20):
                        print("topic word:" + emoji)
                        print("add topic")
                        emojiwords.add(emoji)
                        newemojiwords.add(emoji)
    print(newemojiwords)
    # 将收集到的词写入词库
    # if (emojiwords.__len__() > 0):
    file = open('../data/微博话题词库.txt', 'a')
    for emoji in newemojiwords:
        # f.writelines(emojiwords)
        if (emoji.startswith('#')):
            file.write(emoji + "\n")
        else:
            file.write("#" + emoji + "#\n")
    file.close()


# inputs, outputs = rdf.readcsv('../data/weibo_senti_100k.csv')
# find_emoji_words(inputs)
inputs = []
outputs = []
pd_all = pd.read_csv("../matchOriginData/【测试语料】正负面测试语料（选手用）.csv",
                     encoding='gb18030')
# pd_all = pd.read_csv("../data/weibo_senti_100k.csv",index_col=0)
# print(pd_all)
file = open('../data/验证集评论.txt', 'a')
for index in pd_all.index:
    print(pd_all.loc[index].text)
    inputs.append(pd_all.loc[index].text)
    file.write(str(pd_all.loc[index].text) + "\n")
    # outputs.append(pd_all.loc[index].emotion)

print(inputs)
print("评论加载完毕")
inputs = rdf.readtxt("../data/验证集评论.txt")
find_emoji_words(inputs)
# find_topic_words(inputs)

