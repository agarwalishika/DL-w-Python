import sys
sys.path.append('C:\\Users\\ishik\\le code\\DL in python')

from keras import models
from keras import layers
from keras.datasets import imdb
from keras_preprocessing import sequence
from graph_results import eval_model

# make the data usable
max_features = 10000
maxlen = 500
batch_size = 32

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# create model
network = models.Sequential()
network.add(layers.Embedding(max_features, 32))
# return_sequences is whether you want the entire history or just the last one
network.add(layers.SimpleRNN(32))
# you should mostly stack recurrent layers
# network.add(layers.SimpleRNN(32, return_sequences=True))
# network.add(layers.SimpleRNN(32, return_sequences=True))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = network.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

eval_model(history)
