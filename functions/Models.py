import functions.ReadDataFile as rdf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import text
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold

#词索引数
vocab_size = 3000
#句子最大词数
maxLength = 400
#词向量维度
word_vector_size = 200
#交叉验证方法
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

#得到输入数据的每行词编码
def get_encoded_docs(wordlist):
    trainSentence = []
    for words in wordlist:
        sentence = ""
        for word in words:
            sentence = sentence + word + " "
        trainSentence.append(sentence.strip())
    encoded_docs = [text.one_hot(d, vocab_size,
                                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
                    for d in trainSentence]
    return pad_sequences(encoded_docs, maxlen=maxLength, padding='post')

if __name__ == "__main__":
    inputs, outputs = rdf.readcsv('../data/ChnSentiCorp_htl_all.csv')
    wordlist = rdf.docs_to_wordlist(inputs, rdf.readtxt('../data/中文停用词库.txt'))
    padded_docs = get_encoded_docs(wordlist)
    out_put_array = np.array(outputs)

    scores = []
    for train, test in kfold.split(padded_docs, out_put_array):
        model = Sequential()
        #词嵌入层
        model.add(Embedding(vocab_size, word_vector_size, input_length=maxLength))
        #将输入压为1维数组
        model.add(Flatten())
        #全连接层
        model.add(Dense(1, activation='sigmoid'))
        #模型编译
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        #模型训练
        model.fit(padded_docs[train], out_put_array[train], epochs=10, verbose=0)
        #在验证集上评估
        loss, accuracy = model.evaluate(padded_docs[test], out_put_array[test], verbose=0)
        scores.append(100 * accuracy)
        print(100 * accuracy)
    #打印准确率分布
    print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))




# trainSentence = []
# trainLabel = []
# evalSentence = []
# evalLabel = []
# testSentence = []
# testLabel = []
#
# rate = np.array([0.9, 0.1])
# cumsum_rate = np.cumsum(rate)
#
# for index in pd_all.index:
#     review = pd_all.loc[index].review
#     label = pd_all.loc[index].label
#     random = int(np.searchsorted(cumsum_rate, np.random.rand(1) * 1.0))
#
#     if random == 0:
#         seg_list = jieba.cut(str(review))
#         trainSentence.append(" ".join(seg_list))
#         trainLabel.append(label)
#
#     if random == 1:
#         seg_list = jieba.cut(str(review))
#         testSentence.append(" ".join(seg_list))
#         testLabel.append(label)
#
# vocab_size = 3000
# encoded_docs = [text.one_hot(d, vocab_size,
#                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
#                 for d in trainSentence]
# encoded_test_docs = [text.one_hot(d, vocab_size,
#                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
#                 for d in testSentence]
#
# max_length = 100
# padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
#
# padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_length, padding='post')
#
# trainArray = np.array(trainLabel)
#
# scores = []
#
# for train, test in kfold.split(padded_docs, trainArray):
    # define the model
#     model = Sequential()
#     model.add(Embedding(vocab_size, 200, input_length=max_length))
#     model.add(Flatten())
#     model.add(Dense(1, activation='sigmoid'))
#
#     # compile the model
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#
#     model.fit(padded_docs[train], trainArray[train],
#           epochs=10, verbose=0, callbacks=[TensorBoard(log_dir='./tmp/log')])
#     loss, accuracy = model.evaluate(padded_docs[test], trainArray[test], verbose=0)
#     scores.append(100*accuracy)
#
# print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))




