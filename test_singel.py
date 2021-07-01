from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib.image import imread
import os
import cv2
import link, func

model = load_model(link.model_save)
path = "C:/Users/User/Downloads/Pedestrians/Pedestrian/test_png/"
d = os.listdir(path)

for f in d:
    image_path = path + f
    image = imread(image_path)
    print(image.shape)
    func.testimage(image, model)

