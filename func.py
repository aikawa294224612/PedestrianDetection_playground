from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.image import imread
import numpy as np
import cv2
import link

THRESHOLD = 0.95

def tofloat(data):
    return data.astype('float32')


def printacc(his):
    return his.history['accuracy'][-1]


def printvalacc(his):
    return his.history['val_accuracy'][-1]


def show_train_history(train_history, train, validation, title):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title(title)
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def createModel():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, input_shape=(36, 18, 3), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.2))

    # model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPool2D(pool_size=2))
    # model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    return model


def imageGener(filedir, df):
    gen = ImageDataGenerator(rescale=1.0 / 255.0)
    gen = gen.flow_from_dataframe(df,
                                  directory=filedir,
                                  x_col='file',
                                  y_col='label',
                                  target_size=(18, 36),
                                  class_mode='categorical',
                                  batch_size=32,
                                  color_mode='rgb',
                                  shuffle=False)
    return gen


def confident(prediction):
    if prediction[1] > THRESHOLD:
        return 1
    else:
        return 0


def confident_single(prediction):
    if prediction[0][1] > THRESHOLD:
        print(prediction[0][1])
        return 1
    else:
        return 0


def testimage(iread, model):
    image = cv2.cvtColor(iread, cv2.COLOR_BGR2RGB)
    image = tofloat(image)
    image = cv2.resize(image, (36, 18))
    img_resize = np.reshape(image, (1, 36, 18, 3))

    predictions = model.predict(img_resize)
    print(predictions)
    high_cof = confident_single(predictions)
    # prediction_max = np.argmax(predictions, axis=1)
    # print(prediction_max)
    return high_cof


def testimage_max(iread, model):
    image = cv2.cvtColor(iread, cv2.COLOR_BGR2RGB)
    image = tofloat(image)
    image = cv2.resize(image, (36, 18))
    img_resize = np.reshape(image, (1, 36, 18, 3))

    predictions = model.predict(img_resize)
    print(predictions)
    high_cof = np.argmax(predictions, axis=1)
    return high_cof


def writeandread(sub, path):
    cv2.imwrite(path, sub)
    image = imread(path)
    return image


def getmodel(type):
    return link.choose_model if type == 1 else link.model_save


def addtolist(x, y, w, h, rec_list_before, rec_list_after):
    before = (x, y)
    after = (x + w, y + h)
    rec_list_before.append(before)
    rec_list_after.append(after)
    return rec_list_before, rec_list_after
