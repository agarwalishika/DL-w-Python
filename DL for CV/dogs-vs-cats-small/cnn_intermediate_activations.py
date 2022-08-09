# going to use one of the models that we've already trained to visualize intermediate activations (which is 
# useful for understanding how each layer transforms their input). it provides insight into how an input
# image is decomposed into different filters

from keras import models
from keras import layers
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

model = models.load_model('cats_and_dogs_small_2.h5')
# print(model.summary())

img_path = 'C:\\Users\\ishik\\le code\\DL in python\\dogs-vs-cats-small\\test\\cats\\cat.1700.jpg'
img = load_img(img_path, target_size=(150, 150))
img_tensor = img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0) # takes a (150x150x3) and turns it into a (1x150x150x3) to 
                                                # indicate that there is one image of (150x150x3)
img_tensor /= 255

plt.imshow(img_tensor[0])
plt.show()

# we want to extract feature maps. the below 'activation_model' will
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
for i in range(0, 31):
    plt.matshow(first_layer_activation[0,:,:,i], cmap='viridis')
    plt.show()