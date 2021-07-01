from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras as K
from sklearn.utils import shuffle
import cv2
import link, func

train_path = link.train_path
test_path = link.test_path

train_dir = os.listdir(train_path)
test_dir = os.listdir(test_path)

train_data = []
train_label = []

test_data = []
test_label = []

EPOCHS = 15
BATCH_SIZE = 32

def getdata(listdir, type):
    global train_data
    global train_label

    global test_data
    global test_label

    for f in listdir:
        if type == 'train':
            image = imread(train_path + f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            train_data.append(image)
            if 'ped' in f:
                train_label.append(1)
            else:
                train_label.append(0)
        else:
            image = imread(test_path + f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            test_data.append(image)
            if 'ped' in f:
                test_label.append(1)
            else:
                test_label.append(0)


getdata(train_dir, 'train')
getdata(test_dir, 'test')

train_data = np.array(train_data)
test_data = np.array(test_data)
train_label = np.array(train_label)
test_label = np.array(test_label)

train_data, train_label = shuffle(train_data, train_label)

train_data = func.tofloat(train_data)
test_data = func.tofloat(test_data)

train_label = K.utils.to_categorical(train_label)
test_label = K.utils.to_categorical(test_label)

train_data = np.reshape(train_data, (-1, 36, 18, 3))
test_data = np.reshape(test_data, (-1, 36, 18, 3))

print("Training set:", train_data.shape)
print("Training label set:", train_label.shape)

model = func.createModel()
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

earlystop = EarlyStopping(patience=2)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0001)

history = model.fit(x=train_data,
                    y=train_label,
                    epochs=EPOCHS,
                    validation_split=0.3,
                    shuffle=True
                    # ,callbacks=[earlystop, learning_rate_reduction]
                    )


model.save(link.model_save)
model.save_weights(link.weight_save)

acc = func.printacc(history)
val_acc = func.printvalacc(history)
print("Acc:", acc)
print("Val Acc:", val_acc)

func.show_train_history(history, 'accuracy', 'val_accuracy', 'Train History')
func.show_train_history(history, 'loss', 'val_loss', 'Loss History')

score = model.evaluate(test_data, test_label, batch_size=200)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict(test_data)
predictions = np.argmax(predictions, axis=1)


for i in range(12):
    img = test_data[i]
    plt.subplot(3, 4, i+1)
    plt.imshow(img)
    plt.title(f"Class:{predictions[i]}")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()



