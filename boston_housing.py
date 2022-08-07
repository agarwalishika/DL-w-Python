from venv import create
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def create_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(13,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop',
              loss='mse', # mean-squared error (square of the difference between prediction and actual)
              metrics=['mae']) # mean absolute error (abs value of actual - predicted)
    
    return model

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for p in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + p * (1 - factor)) #exponential moving average
        else: smoothed_points.append(p)
    
    return smoothed_points


(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# note: train_data is a 404 x 13 array

# for continuous data, you'd want to subtract the mean from all the data points so that 
# "the feature is centered around 0 and has a unit standard deviation".
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

# Build the model

# we are going to do k-fold cross validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500 #was 100, but we want to log the mae scores
all_scores = []
all_mae_histories =[]

for i in range(k):
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_labels = train_labels[i * num_val_samples: (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis = 0
    )


    partial_train_labels = np.concatenate(
        [train_labels[:i * num_val_samples],
        train_labels[(i + 1) * num_val_samples:]],
        axis = 0
    )

    model = create_model()
    history = model.fit(partial_train_data,
                        partial_train_labels,
                        validation_data=(val_data, val_labels),
                        epochs = num_epochs,
                        batch_size=1)

    all_mae_histories.append(history.history['val_mae'])
    val_mse, val_mae = model.evaluate(val_data, val_labels)
    all_scores.append(val_mae)

    

print(all_scores)

# plot the mae
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
smooth_avg_mae = smooth_curve(average_mae_history[10:]) # the first 10 points have an exaggerated loss, so emit those 

plt.plot(range(1, len(smooth_avg_mae) + 1), smooth_avg_mae)
plt.xlabel('Epochs')
plt.ylabel('Smoothed Validation MAE')
plt.show()
