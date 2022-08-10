import numpy as np

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features))
state_t = np.random.random(output_features)

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, input_features))
b = np.random.random((output_features))

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, input_t) + b) # 1 if positive and -1 if negative
    successive_outputs.append(output_t)
    state_t = output_t

final_output = np.concatenate(successive_outputs, axis=0)

print(final_output)