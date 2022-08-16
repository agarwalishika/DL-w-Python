import sys
sys.path.append('C:\\Users\\ishik\\le code\\DL in python')

import os
import numpy as np
import matplotlib.pylab as plt
from keras import layers
from keras import models
from graph_results import eval_loss

'''
method creates a generator based on the following information:
data: normalized data
lookback: number of timesteps that the input data should go back
delay: the number of timesteps in the future the target should be in
min_/max_index: indices of 'data' that can be used for sampling
shuffle: boolean to shuffle the data or not
batch_size: number of samples per batch
step: the period in timesteps at which you sample the data (6 means taking one datapoint per hour)
'''
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback

    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

def evaluate_naive_method(val_steps, val_gen):
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))


data_dir = 'C:\\Users\\ishik\\le code\\DL in python\\DL for text\\jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

# remove the date
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    float_data[i, :] = [float(x) for x in line.split(',')[1:]]

'''temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)
plt.show()
'''

# make data usable
train_size = 200000

mean = float_data[:train_size].mean(axis=0)
float_data -= mean
std = float_data[:train_size].std(axis=0)
float_data /= std

lookback = 720
step = 3
delay = 144
batch_size = 128

train_gen = generator(
    data=float_data,
    lookback=lookback,
    delay=delay,
    min_index=0,
    max_index=200000,
    shuffle=True,
    step=step,
)

val_gen = generator(
    data=float_data,
    lookback=lookback,
    delay=delay,
    min_index=200001,
    max_index=300000,
    shuffle=True,
    step=step,
)

test_gen = generator(
    data=float_data,
    lookback=lookback,
    delay=delay,
    min_index=300001,
    max_index=None,
    step=step,
    batch_size=batch_size
)

val_steps = (300000 - 200001 - lookback) // 128
test_steps = (len(float_data) - 300001 - lookback) // 128

network = models.Sequential()
network.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])))
network.add(layers.MaxPooling1D(3))
network.add(layers.Conv1D(32, 5, activation='relu'))
network.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
network.add(layers.Dense(1))

network.compile(
    optimizer='rmsprop',
    loss='mae'
)

history = network.fit_generator(
    train_gen,
    steps_per_epoch=500,
    epochs=20,
    validation_data=val_gen,
    validation_steps=val_steps
)

eval_loss(history)