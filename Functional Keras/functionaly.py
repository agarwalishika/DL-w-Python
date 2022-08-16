from keras import Input, layers

input_tensor = Input(shape=(32,))
dense = layers.Dense(activation='relu')
output_tensor = dense(input_tensor)

# to add layers in between, you just store the output of the layers.LAYER_TYPE and then
# input the previous output from the previous layer like this:
input_tensor = Input(shape=(32,))
dense = layers.Dense(32, activation='relu')(input_tensor)
dense = layers.Dense(32, activation='relu')(dense)
output_tensor = layers.Dense(10, activation='softmax')(dense)


# compiling/training and evaluating are the exact same