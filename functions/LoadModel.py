import numpy as np
import gensim as gs
from keras.layers.embeddings import Embedding
import functions.ReadDataFile as rdf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.text import text
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold

model = gs.models.Word2Vec.load("../data/word2vecLarge.model")
vocab_list = [word for word, Vocab in model.wv.vocab.items()]
word_index = {" ": 0}
word_vector = {}
embeddings_matrix = np.zeros((len(vocab_list) + 1, model.vector_size))
print(len(vocab_list))
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

for i in range(len(vocab_list)):
    word = vocab_list[i]
    word_index[word] = i + 1
    word_vector[word] = model.wv[word]
    embeddings_matrix[i + 1] = model.wv[word]

#得到输入数据的每行词编码
def get_encoded_docs(wordlist):
    trainSentence = []
    for words in wordlist:
        sentence = []
        for word in words:
            try:
                sentence.append(word_index[word])
            except:
                sentence.append(0)
        trainSentence.append(sentence)
    return pad_sequences(trainSentence, maxlen=135)

#模型
def init_model():
    model = Sequential()
    # 词嵌入层
    embedding_layer = Embedding(len(embeddings_matrix), 300, weights=[embeddings_matrix],
                                input_length=135, trainable=False)
    model.add(embedding_layer)
    # 将输入压为1维数组
    model.add(Flatten())
    # 全连接层
    model.add(Dense(1, activation='sigmoid'))
    # 模型编译
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model

if __name__ == "__main__":
    inputs, outputs = rdf.readcsv('../data/weibo_senti_100k.csv')
    wordlist = rdf.docs_to_wordlist(inputs, rdf.readtxt('../data/中文停用词库.txt'))
    #输入向量
    padded_docs = get_encoded_docs(wordlist)
    # 标签集合
    out_put_array = np.array(outputs)
    # 模型打分
    scores = []

    for train, test in kfold.split(padded_docs, out_put_array):
        # 使用基准模型
        model = init_model()
        # 模型训练
        model.fit(padded_docs[train], out_put_array[train], epochs=10, verbose=0)
        # 在验证集上评估
        loss, accuracy = model.evaluate(padded_docs[test], out_put_array[test], verbose=0)
        scores.append(100 * accuracy)
        print(100 * accuracy)

    # 打印准确率分布
    print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))






