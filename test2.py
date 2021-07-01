from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
from tensorflow.keras.models import load_model
import tensorflow.keras as K
import cv2
import link, func

test_path = link.test_path
test_dir = os.listdir(test_path)
test_data = []
test_label = []


def getdata(listdir):
    global test_data
    global test_label

    for f in listdir:
        image = imread(test_path + f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        test_data.append(image)
        if 'ped' in f:
            test_label.append(1)
        else:
            test_label.append(0)


getdata(test_dir)
test_data = np.array(test_data)
test_label = np.array(test_label)
test_data = func.tofloat(test_data)

test_label = K.utils.to_categorical(test_label)
test_data = np.reshape(test_data, (-1, 36, 18, 3))

model = load_model(link.model_save)

predictions = model.predict(test_data)
print(predictions)
predictions = np.argmax(predictions, axis=1)
print(predictions)

for i in range(12):
    img = test_data[i]
    plt.subplot(3, 4, i+1)
    plt.imshow(img)
    plt.title(f"Class:{predictions[i]}")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()



