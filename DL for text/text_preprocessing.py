import sys
sys.path.append('C:\\Users\\ishik\\le code\\DL in python')

from keras import models
from keras import layers
import os
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from graph_results import eval_model

def get_dict_value(dict, key):
    if key not in dict.keys():
        return None
    return dict[key]

imdb_dir = 'C:\\Users\\ishik\\le code\\DL in python\\DL for text\\aclImdb'

# read the data

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(imdb_dir, label_type)
    for f in os.listdir(dir_name):
        if f[-4:] == '.txt':
            f = open(os.path.join(dir_name, f), encoding="utf8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)


# make data usable
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)

# shuffling the data
indicies = np.arange(data.shape[0])
np.random.shuffle(indicies)
data = data[indicies]
labels = labels[indicies]

# split data into training and validation
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples : training_samples + validation_samples]
y_val = labels[training_samples : training_samples + validation_samples]


# load the pretrained glove model
glove_dir = 'C:\\Users\\ishik\\le code\\DL in python\\glove.42B.300d.txt'
embeddings_index = {}
f = open(glove_dir, encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# build an embedding matrix that we can add to the Embedding layer
embedding_dim = 300 # number of coefficients for each word
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = get_dict_value(embeddings_index, word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# create the model
network = models.Sequential()
network.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
network.add(layers.Flatten())
network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

# input the embedding matrix
# if you 
# network.layers[0].set_weights([embedding_matrix])
# network.layers[0].trainable = False

network.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics='accuracy')

history = network.fit(x_train, y_train,
                      epochs=10,
                      batch_size=32,
                      validation_data=(x_val, y_val))

network.save_weights('pre_trained_glove_model.h5')

eval_model(history)