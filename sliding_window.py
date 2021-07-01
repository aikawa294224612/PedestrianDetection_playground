import cv2
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib.image import imread
import link, func

test_path = link.testing_path
image_path = test_path + "16m_11s_142099u_resize.png"
model = func.getmodel(1)
print(model)

model = load_model(model)

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

print(img.shape)

w = int(18*1.2)
h = int(36*1.2)

img_w = img.shape[1]
img_h = img.shape[0]
w_times = img_w - w + 1
h_times = img_h - h + 1


i = 1
last_list = list(np.zeros(2))
rec_list_before = []
rec_list_after = []
for ht in range(h_times):
    for t in range(w_times):
        sub_img = img[ht:ht + h, t:t + w]
        image_path = link.test_output + str(i) + '.png'
        # sub_img_resize = cv2.resize(sub_img, (36, 18))
        # predict_max = func.testimage(sub_img_resize, model)

        image = func.writeandread(sub_img, image_path)
        i += 1
        predict_max = func.testimage(image, model)

        if predict_max == 1:
            if last_list.count(1) == 2:
                before = (t, ht)
                after = (t + w, ht + h)
                rec_list_before.append(before)
                rec_list_after.append(after)
            last_list.append(1)
            last_list.pop(0)
        else:
            last_list.append(0)
            last_list.pop(0)



img_copy = np.copy(img)
for i in range(len(rec_list_before)):
    cv2.rectangle(img_copy, pt1=rec_list_before[i], pt2=rec_list_after[i], color=(255, 0, 0), thickness=1)
plt.imshow(img_copy)
plt.show()
