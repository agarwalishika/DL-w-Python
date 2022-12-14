from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

img_path = 'C:\\Users\\ishik\\le code\\DL in python\\elephants.jpg'

model = VGG16(weights='imagenet')

img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# top categories that correlate to the image: top most category is "African Elephant"
preds = model.predict(x)
print(decode_predictions(preds, top=3)[0])

# now, let's see which parts of the picture are the most "African Elephant"-like
ind = np.argmax(preds[0]) # gets the index of the max value ("African Elephant")

african_elephant_output = model.output[:, ind]
last_conv_layer = model.get_layer('block5_conv3')

grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()