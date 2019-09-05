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
#TODO
vocab_size = 3000
#句子最大词数
#TODO
maxLength = 400
#词向量维度
#TODO
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

#基准模型
def init_base_line_model():
    model = Sequential()
    # 词嵌入层
    model.add(Embedding(vocab_size, word_vector_size, input_length=maxLength))
    # 将输入压为1维数组
    model.add(Flatten())
    # 全连接层
    model.add(Dense(1, activation='sigmoid'))
    # 模型编译
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model

if __name__ == "__main__":
    inputs, outputs = rdf.readcsv('../data/ChnSentiCorp_htl_all.csv')
    wordlist = rdf.docs_to_wordlist(inputs, rdf.readtxt('../data/中文停用词库.txt'), rdf.readtxt('../data/白名单词库.txt'))
    #输入向量
    padded_docs = get_encoded_docs(wordlist)
    #标签集合
    out_put_array = np.array(outputs)
    #模型打分
    scores = []
    # 使用基准模型
    model = init_base_line_model()

    for train, test in kfold.split(padded_docs, out_put_array):
        #模型训练
        model.fit(padded_docs[train], out_put_array[train], epochs=10, verbose=0)
        #在验证集上评估
        loss, accuracy = model.evaluate(padded_docs[test], out_put_array[test], verbose=0)
        scores.append(100 * accuracy)
        print(100 * accuracy)

    #打印准确率分布
    print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
