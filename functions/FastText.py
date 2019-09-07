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

def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        new_list = new_list.tolist()
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

def add_ngram_doc(padded_docs,vocab_size):
    ngram_set = set()
    for input_list in padded_docs:
        for i in range(2, 3):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    start_index = vocab_size + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}
    max_features = np.max(list(indice_token.keys())) + 1

    new_padded = add_ngram(padded_docs, token_indice)
    new_padded_docs = pad_sequences(new_padded, maxlen=79)
    return new_padded_docs, max_features

#词索引数
vocab_size = 2000
#句子最大词数
maxLength = 40
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

#基准模型
def init_base_line_model(max_features):
    model = Sequential()
    # 词嵌入层
    model.add(Embedding(max_features, word_vector_size, input_length=79))
    # 将输入压为1维数组
    model.add(Flatten())
    # 全连接层
    model.add(Dense(1, activation='sigmoid'))
    # 模型编译
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model

if __name__ == "__main__":
    inputs, outputs = rdf.readcsv('../data/ChnSentiCorp_htl_all.csv')
    wordlist = rdf.docs_to_wordlist(inputs, rdf.readtxt('../data/中文停用词库.txt'))
    #输入向量
    padded_docs = get_encoded_docs(wordlist)
    new_padded_docs, max_features = add_ngram_doc(padded_docs, vocab_size)
    #标签集合
    out_put_array = np.array(outputs)
    #模型打分
    scores = []

    for train, test in kfold.split(new_padded_docs, out_put_array):
        # 使用基准模型
        model = init_base_line_model(max_features)
        print(max_features)
        print(model.summary())
        #模型训练
        model.fit(new_padded_docs[train], out_put_array[train], epochs=10, verbose=0)
        #在验证集上评估
        loss, accuracy = model.evaluate(new_padded_docs[test], out_put_array[test], verbose=0)
        scores.append(100 * accuracy)
        print(100 * accuracy)

    #打印准确率分布
    print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))