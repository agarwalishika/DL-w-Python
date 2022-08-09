# going to use one of the models that we've already trained to visualize intermediate activations (which is 
# useful for understanding how each layer transforms their input). it provides insight into how an input
# image is decomposed into different filters

from keras import models
from keras import layers
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# NOTE: THE FUNCTION visualize_channels_in_activation() HAS BEEN COPY-PASTED
def visualize_channels_in_activation():
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name) 
    
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features # images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
        
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

model = models.load_model('cats_and_dogs_small_2.h5')
# print(model.summary())

img_path = 'C:\\Users\\ishik\\le code\\DL in python\\dogs-vs-cats-small\\test\\cats\\cat.1700.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
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
for i in range(32):
    plt.matshow(first_layer_activation[0,:,:,i], cmap='viridis')