import keras
import numpy as np
from keras import layers
from keras import models
import random
import sys

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# make data usable
maxlen = 60
step = 3

path = keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()

sentences = []
next_char = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_char.append(text[i + maxlen])

chars = sorted(list(set(text)))
char_indices = dict((char, chars.index(char)) for char in chars)

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_char[i]]] = 1

# create model

network = models.Sequential()
network.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
network.add(layers.Dense(len(chars), activation='softmax'))

network.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.01),
    loss='categorical_crossentropy'
)

# train the model and sample from it
for epoch in range(1, 60):
    print(f'\nepoch {epoch}')
    network.fit(
        x, y,
        batch_size=128,
        epochs=1
    )
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print(f'\ngenerating with seed: {generated_text}')

    for temperature in [0.2, 0.5, 1, 1.2]:
        print(f'\ntemperature: {temperature}')
        sys.stdout.write(generated_text)

        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1
            
            preds = network.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)