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

# build network
network = models.Sequential()
network.add(layers.Embedding(max_features, 32))
network.add(layers.Bidirectional(layers.LSTM(32)))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# train model
history = network.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# evaluate model
eval_model(history)