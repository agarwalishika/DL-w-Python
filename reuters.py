from keras.datasets import reuters
from keras.utils import to_categorical
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

num_labels = 46
num_words = 10000
num_epochs = 9

def vectorize_array(array, dimension):
    results = np.zeros((len(array), dimension))
    for i, sequence in enumerate(array):
        results[i, sequence] = 1
    return results

# Make data usable
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=num_words)

x_train = vectorize_array(train_data, num_words)
x_test = vectorize_array(test_data, num_words)

#y_train = vectorize_array(train_labels, num_labels)
y_train = to_categorical(train_labels)
#y_test = vectorize_array(test_labels, num_labels)
y_test = to_categorical(test_labels)


# Build model
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(num_words,)))
# if you want to add a dropout layer, then this is how you do it:
# model.add(layers.Dropout(0.5))
# from what I can tell, I think this only allow 50% of the values to reach from one layer to another.
# usually, dropout values are between 0.2 and 0.5
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(36, activation='relu')) -> greatly reduces accuracy lol
model.add(layers.Dense(num_labels, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# set apart 1000 samples for validation
x_validation = x_train[:1000]
partial_x_train = x_train[1000:]

y_validation = y_train[:1000]
partial_y_train = y_train[1000:]

# Train the model
history = model.fit(partial_x_train,
          partial_y_train,
          epochs=num_epochs,
          batch_size=512,
          validation_data=(x_validation, y_validation))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"test acc:{test_acc}, test_loss={test_loss}")

# plotting training/validation loss
history_dict = history.history
loss_vals = history_dict['loss']
validation_loss_vals = history_dict['val_loss']
epochs = range(1, num_epochs + 1)

plt.plot(epochs, loss_vals, 'bo', label='Training loss')
plt.plot(epochs, validation_loss_vals, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plotting training/validation accuracy
plt.clf()
acc_vals = history_dict['accuracy']
validation_acc_vals = history_dict['val_accuracy']

plt.plot(epochs, acc_vals, 'bo', label='Training accuracy')
plt.plot(epochs, validation_acc_vals, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()