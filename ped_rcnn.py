import cv2
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import link, func

test_path = link.testing_path
image_path = test_path + "15m_23s_264082u.png"
last_list = list(np.zeros(2))
rec_list_before = []
rec_list_after = []

model = func.getmodel(1)
print(model)
model = load_model(model)

im = cv2.imread(image_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# https://blog.gtwang.org/programming/selective-search-for-object-detection/
# selective-search
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(im)
ss.switchToSelectiveSearchFast()

# 使用精準模式（速度較慢）
# ss.switchToSelectiveSearchQuality()

rects = ss.process()
print(rects.shape)
print('候選區域總數量： {}'.format(len(rects)))

numShowRects = rects.shape[0]

imOut = im.copy()
for i, rect in enumerate(rects):
    if i < numShowRects:
        x, y, w, h = rect
        sub = imOut[y: y+h, x: x+w]
        # sun_resize = cv2.resize(sub, (18, 36))

        image_path = link.test_output + str(i) + '.png'
        image = func.writeandread(sub, image_path)

        predict_max = func.testimage(image, model)

        if predict_max == 1:
            rec_list_before, rec_list_after = func.addtolist(x, y, w, h, rec_list_before, rec_list_after)
    else:
        break

print(rec_list_before)
img_copy = np.copy(im)
for i in range(len(rec_list_before)):
    cv2.rectangle(img_copy, rec_list_before[i], rec_list_after[i], (0, 255, 0), 1, cv2.LINE_AA)
plt.imshow(img_copy)
plt.show()
