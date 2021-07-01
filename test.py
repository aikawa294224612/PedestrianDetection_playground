from matplotlib import pyplot as plt
from matplotlib.image import imread
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import link, func


# type = 0: train, 1: test
def dataframe(lisdir):
    global test_df

    label = 'nor'
    for f in lisdir:
        if 'ped' in f:
            label = 'ped'
        diction = {"file": f, "label": label}
        test_df = test_df.append(diction, ignore_index=True)


model = load_model(link.model_save)

test_path = link.test_path
test_dir = os.listdir(test_path)
test_df = pd.DataFrame()

dataframe(test_dir)
test_gen = func.imageGener(test_path, test_df)

predictions = model.predict(test_gen)
print(predictions)
prediction_max = np.argmax(predictions, axis=1)
print(prediction_max)

for i, file in enumerate(test_df['file'][:12]):
    img = imread(test_path + file)
    plt.subplot(3, 4, i+1)
    plt.imshow(img)
    plt.title(f"Class:{func.confident(predictions[i])}")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()

