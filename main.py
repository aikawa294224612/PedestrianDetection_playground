from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.image import imread
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import link

train_path = link.train_path
test_path = link.test_path

train_dir = os.listdir(train_path)
test_dir = os.listdir(test_path)

train_df = pd.DataFrame()
test_df = pd.DataFrame()

EPOCHS = 30
BATCH_SIZE = 32


# type = 0: train, 1: test
def getdataframe(lisdir, type):
    global train_df
    global test_df

    label = 'nor'
    for f in lisdir:
        if 'ped' in f:
            label = 'ped'
        diction = {"file": f, "label": label}
        if type == 0:
            train_df = train_df.append(diction, ignore_index=True)
        else:
            test_df = test_df.append(diction, ignore_index=True)


def imageGener(filedir, df):
    gen = ImageDataGenerator(rescale=1.0 / 255.0)
    gen = gen.flow_from_dataframe(df,
                                  directory=filedir,
                                  x_col='file',
                                  y_col='label',
                                  target_size=(18, 36),
                                  class_mode='categorical',
                                  batch_size=BATCH_SIZE,
                                  color_mode='rgb',
                                  shuffle=False)
    return gen


def showImage(image_gen):
    for x_gens, y_gens in image_gen:
        print(x_gens.shape)
        x_gen_shape = x_gens.shape[1:]
        i = 0
        for sample_img, sample_class in zip(x_gens, y_gens):
            plt.subplot(2,4,i+1)
            plt.title(f'Class:{np.argmax(sample_class)}')
            plt.axis('off')
            plt.imshow(sample_img)

            i += 1

            if i >= 8:
                break
        break

    plt.show()


def createModel():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, input_shape=(18, 36, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    return model


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


getdataframe(train_dir, 0)
getdataframe(test_dir, 1)


train_df, valid_df = train_test_split(train_df, test_size=0.15)
print("Training set:", train_df.shape)
print("Validation set:", valid_df.shape)

print(valid_df)

train_gen = imageGener(train_path, train_df)
valid_gen = imageGener(train_path, valid_df)
test_gen = imageGener(test_path, test_df)

model = createModel()
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

history = model.fit(train_gen,
                    steps_per_epoch=len(train_df)//BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=valid_gen,
                    validation_steps=len(valid_df)//BATCH_SIZE
                    )

model.save(link.model_save)
model.save_weights(link.weight_save)

acc = printacc(history)
val_acc = printvalacc(history)
print("Acc:", acc)
print("Val Acc:", val_acc)

show_train_history(history, 'accuracy', 'val_accuracy', 'Train History')
show_train_history(history, 'loss', 'val_loss', 'Loss History')

predictions = model.predict(test_gen)
predictions = np.argmax(predictions, axis=1)

for i, file in enumerate(test_df['file'][:12]):
    img = imread(test_path + file)
    plt.subplot(3, 4, i+1)
    plt.imshow(img)
    plt.title(f"Class:{predictions[i]}")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()



