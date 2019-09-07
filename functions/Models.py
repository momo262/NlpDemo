import functions.ReadDataFile as rdf
import functions.TextCnn as tcnn
import functions.LSTM as lstm
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import text
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

#词索引数
vocab_size = 6000
#句子最大词数
maxLength = 135
#词向量维度
word_vector_size = 300
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

def toOneHot(out_put):
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit([[0], [1]])
    out_put2 = []
    for out in out_put:
        out_put1 = []
        out_put1.append(out)
        out_put2.append(out_put1)
    result = enc.transform(out_put2)
    return result.toarray()

if __name__ == "__main__":
    inputs, outputs = rdf.readcsv('../data/weibo_senti_100k.csv')
    wordlist = rdf.docs_to_wordlist(inputs, rdf.readtxt('../data/中文停用词库.txt'))
    #输入向量
    padded_docs = get_encoded_docs(wordlist)
    #标签集合
    out_put_array = np.array(outputs)
    #模型打分
    scores = []
    # 使用基准模型
    # model = init_base_line_model()

    for train, test in kfold.split(padded_docs, out_put_array):
        # model = tcnn.init_cnn_model()
        model = lstm.init_lstm_att_model()
        # model = lstm.init_lstm_model()
        #模型训练
        model.fit(padded_docs[train], toOneHot(out_put_array[train]), epochs=4, verbose=0)
        #在验证集上评估
        loss, accuracy = model.evaluate(padded_docs[test], toOneHot(out_put_array[test]),
                                        verbose=0, callbacks=[TensorBoard(log_dir='./tmp/log')])
        scores.append(100 * accuracy)
        print(100 * accuracy)

    #打印准确率分布
    print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
