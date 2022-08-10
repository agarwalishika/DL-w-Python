from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Embedding(10000, 32))
# return_sequences is whether you want the entire history or just the last one
network.add(layers.SimpleRNN(32, return_sequences=True))
# you should mostly stack recurrent layers
network.add(layers.SimpleRNN(32, return_sequences=True))
network.add(layers.SimpleRNN(32, return_sequences=True))
