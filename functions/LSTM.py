from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding ,Dense ,merge,Input,LSTM,Permute,Softmax,Lambda,Flatten,CuDNNGRU
from keras import Model
import keras.backend as K
from keras.utils import to_categorical
import os

#词索引数
vocab_size = 6000
#句子最大词数
maxLength = 135
#词向量维度
word_vector_size = 300

filters = 250
kernel_size = 3
hidden_dims = 250

def init_lstm_model():
    model = Sequential()
    model.add(Embedding(vocab_size, word_vector_size, input_length=maxLength))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def init_lstm_att_model():
    input_ = Input(shape=(maxLength,))
    words = Embedding(vocab_size, word_vector_size, input_length=maxLength)(input_)
    sen = Bidirectional(LSTM(128,return_sequences=True)) (words)

    # attention
    attention_pre = Dense(1, name='attention_vec')(sen)  # [b_size,maxlen,1]
    attention_probs = Softmax()(attention_pre)  # [b_size,maxlen,1]
    attention_mul = Lambda(lambda x: x[0] * x[1])([attention_probs, sen])

    output = Flatten()(attention_mul)
    output = Dense(32, activation="relu")(output)
    output = Dense(2, activation='softmax')(output)
    model = Model(inputs=input_, outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    return model
    print(model.summary())

