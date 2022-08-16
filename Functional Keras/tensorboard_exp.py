# same as DL for text\convnet.py
import keras
from keras.datasets import imdb
from keras_preprocessing import sequence
from keras import models
from keras import layers

max_features = 2000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

network = models.Sequential()
network.add(layers.Embedding(max_features, 128, input_length=max_len))
network.add(layers.Conv1D(32, 7, activation='relu'))
network.add(layers.MaxPooling1D(5))
network.add(layers.Conv1D(32, 7, activation='relu'))
network.add(layers.GlobalMaxPooling1D())
network.add(layers.Dense(1))

network.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='log_dir',
        histogram_freq=1,
        embeddings_freq=1,
    )
]

# cnns are a cheaper alternative to rnns for word sentiment classification
history = network.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128, 
    validation_split=0.2,
    callbacks=callbacks
)
