
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import tensorflow as tf
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
import keras
from keras.engine.topology import Layer
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras import backend as K
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text
from keras.callbacks import ModelCheckpoint
import json
import sklearn
from sklearn import preprocessing as skpp


# In[2]:

data = pd.read_csv('./dataset/cleaned_lyrics.csv')


# In[4]:

genres = data['genre'].unique()
data['genre_id'] = data.groupby(['genre']).ngroup()

mappings = data[['genre', 'genre_id']].drop_duplicates()
map_list = [(genre_id, genre) for genre, genre_id in mappings.values]
map_list.sort()
map_list

data_subset = data[['genre_id', 'genre', 'lyrics']]


# In[5]:

numpy_data = data['lyrics'].values
max_words = 30000

# create a new Tokenizer
tokenizer = text.Tokenizer(num_words=max_words, oov_token='<UNK>')
# feed our song lyrics to the Tokenizer
tokenizer.fit_on_texts(numpy_data)

# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index

with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)
    
tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= max_words} # <= because tokenizer is 1 indexed
tokenizer.word_index[tokenizer.oov_token] = max_words + 1
indexed_data = tokenizer.texts_to_sequences(numpy_data)
indexed_data = np.array(indexed_data)

label_encoder = skpp.LabelEncoder()
indexed_labels = np.array(label_encoder.fit_transform(data['genre'].values))
#label_encoder.inverse_transform(np.array([10, 8])) #to get original genre text back

num_test = 30000

#shuffle data before splitting off test set
random_indexes = np.random.permutation(len(indexed_labels))
indexed_data = indexed_data[random_indexes]
indexed_labels = indexed_labels[random_indexes]

X_train = indexed_data[:-num_test]
y_train = indexed_labels[:-num_test]
X_test  = indexed_data[-num_test:]
y_test  = indexed_labels[-num_test:]

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

num_words = max_words + 2
# truncate and pad input sequences
max_review_length = 600

X_train_padded = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test_padded = sequence.pad_sequences(X_test, maxlen=max_review_length)


# In[7]:

# Implementation from https://github.com/Kyubyong/transformer/blob/master/modules.py
class PositionalEncoding(Layer):
    def __init__(self, **kwargs):
        super(PositionalEncoding,self).__init__(**kwargs)

    def build(self, input_shape):
        super(PositionalEncoding,self).build(input_shape)

    def call(self, x, mask=None):
        _, T, E = x.get_shape().as_list()
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [tf.shape(x)[0], 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/E) for i in range(E)]
            for pos in range(T)], dtype=np.float32)

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
        return tf.add(outputs, x)

    def get_output_shape_for(self, input_shape):
        return input_shape
    
    def compute_output_shape(self, input_shape):
        return input_shape

def multi_head_attention(x, i, num_heads=4):
    E = embedding_vector_length
    queries = Dense(E, activation='relu')(x)
    keys = Dense(E, activation='relu')(x)
    values = Dense(E, activation='relu')(x)

    # Split and concat
    concat = lambda x: tf.concat(tf.split(x, num_heads, axis=2), axis=0)
    Q_ = Lambda(concat, name="q_reshape_{}".format(i))(queries)
    K_ = Lambda(concat, name="k_reshape_{}".format(i))(keys)
    V_ = Lambda(concat, name="v_reshape_{}".format(i))(values)

    # Multiplication
    matmul = lambda x: tf.matmul(x[0], tf.transpose(x[1], (0, 2, 1)))
    # permute_k = Permute((2, 1))(K_)
    # outputs = K.batch_dot(Q_, permute_k) # (h*N, T_q, T_k)
    outputs = Lambda(matmul, name="q_k_matmul_{}".format(i))([Q_, K_])

    # Scale
    divide = lambda x: x / (K_.get_shape().as_list()[-1] ** 0.5)
    outputs = Lambda(divide, name="divide_{}".format(i))(outputs)

    # Softmax
    softmax = lambda x: tf.nn.softmax(x)
    outputs = Lambda(softmax, name="softmax_{}".format(i))(outputs)

    # Dropouts
    outputs = Dropout(0.1)(outputs)

    # Weighted sum
    matmul2 = lambda x: tf.matmul(x[0], x[1])
    outputs = Lambda(matmul2, name="o_v_matmul_{}".format(i))([outputs, V_])

    # outputs = K.batch_dot(outputs, V_) # ( h*N, T_q, C/h)

    # Restore shape
    concat2 = lambda x: tf.concat(tf.split(x, num_heads, axis=0), axis=2)
    outputs = Lambda(concat2, name="o_reshape_{}".format(i))(outputs) # (N, T_q, C)

    return Add()([outputs, x])


# In[8]:

embedding_vector_length = 100

inputs = Input(shape=(max_review_length,))

embeds = Embedding(num_words, embedding_vector_length, input_length=max_review_length)(inputs)
transformer_input = PositionalEncoding()(embeds)

for i in range(1):
    multi_head = multi_head_attention(transformer_input, i)
    norm = BatchNormalization()(multi_head)
    conv1 = Conv1D(400, 1, activation='relu')(norm)
    conv2 = Conv1D(100, 1)(conv1)
    res = Add()([norm, conv2])
    transformer_input = BatchNormalization()(res)

pooling = GlobalMaxPooling1D()(transformer_input)
outputs = Dense(11, activation='softmax')(pooling)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train_padded, y_train, epochs=3, batch_size=16)


# In[24]:

# Final evaluation of the model
scores = model.evaluate(X_test_padded, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[23]:

model.save_weights('1-transformer-1-epoch-weights_1layer.h5')
