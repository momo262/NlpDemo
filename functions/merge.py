from keras.layers import Dense, Concatenate, Input
from keras.models import Model
import tensorflow as tf
import jieba
import jieba.analyse
import jieba.posseg

# left_input = Input(shape=(None, 100))
# left_model = Dense(8, activation='relu')(left_input)
#
# right_input = Input(shape=(None, 100))
# right_model = Dense(8, activation='relu')(right_input)
#
# conca = Concatenate()([left_model, right_model])
# conca_outputs = Dense(3, activation='softmax')(conca)
#
# model = Model(inputs=[left_input, right_input], outputs=conca_outputs)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

words = jieba.cut(str("她用锁锁住了自己的心灵"))
for w in words:
    print(w)

sentence_seged = jieba.posseg.cut("她用锁锁住了自己的心灵")
outstr = ''
for x in sentence_seged:
    outstr += "{}/{},".format(x.word, x.flag)

print(outstr)

# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
# print(tf.concat([t1, t2], 0))