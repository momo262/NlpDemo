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
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers import multiply

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
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, inputs.shape[1]))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(inputs.shape[1], activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs])
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

def init_lstm_att_model():
    input_ = Input(shape=(maxLength,))
    words = Embedding(vocab_size, word_vector_size, input_length=maxLength)(input_)
    sen = Bidirectional(LSTM(64,return_sequences=True)) (words)
    sen2 = Dropout(0.5)(sen)
    # attention_mul = attention_3d_block(sen2)
    # attention
    attention_pre = Dense(1, name='attention_vec')(sen2)  # [b_size,maxlen,1]
    attention_probs = Softmax()(attention_pre)  # [b_size,maxlen,1]
    attention_mul = Lambda(lambda x: x[0] * x[1])([attention_probs, sen2])

    output = Flatten()(attention_mul)
    output = Dropout(0.5)(output)
    output = Dense(2, activation='softmax')(output)
    model = Model(inputs=input_, outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    print(model.summary())
    return model

def lstm_cnn():
    model = Sequential()
    model.add(Embedding(vocab_size, word_vector_size, input_length=maxLength))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(70))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model


init_lstm_att_model()

