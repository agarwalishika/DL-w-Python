import sys
sys.path.append('C:\\Users\\ishik\\le code\\DL in python')

from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences
from keras import layers
from keras import models
from graph_results import eval_model

maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

network = models.Sequential()
network.add(layers.Embedding(10000, 8, input_length=maxlen))
network.add(layers.Flatten())
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = network.fit(x_train, y_train,
                      epochs=10,
                      batch_size=32,
                      validation_split=0.2)

eval_model(history)


