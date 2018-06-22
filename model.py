# read from driving data csv file
import csv
from random import shuffle

import cv2
import numpy as np
import sklearn

lines = []
with open("data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines.remove(lines[0])

#split the training and validation data

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# Define generator


def generator(lines, batch_size=256):
    num_lines = len(lines)
    while 1: # Loop forever so the generator never terminates
        shuffle(lines)
        for offset in range(0, num_lines, batch_size):
            batch_samples = lines[offset:offset+batch_size]

            batch_images = []
            angles = []

            for batch_sample in batch_samples:
                for i in range(3):
                    image_path = "./data/" + batch_sample[i]
                    image_path = ''.join(image_path.split())
                    center_image = cv2.imread(image_path)
                    batch_images.append(center_image)

                correction = 0.2
                center_angle = float(batch_sample[3])
                angles.append(center_angle)
                angles.append(center_angle + correction)
                angles.append(center_angle - correction)

            # trim image to only see section with road
            X_train = np.array(batch_images)
            y_train = np.array(angles)

            # adding all three images

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(batch_images, angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                flipped_image = cv2.flip(image, 1)
                flipped_measurement = measurement * -1.0
                augmented_images.append(flipped_image)
                augmented_measurements.append(flipped_measurement)

            #print("all ok")

            x_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            print(x_train.shape)

            yield sklearn.utils.shuffle(x_train, y_train)




# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=256)
validation_generator = generator(validation_samples, batch_size=256)


# using keras construct model

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160, 320, 3), output_shape=(160,320,3)))

model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#history_object = model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose=1)


history_object = model.fit_generator(train_generator,samples_per_epoch=len(train_samples),
         nb_epoch=3,validation_data=validation_generator,nb_val_samples=len(validation_samples))


print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model_final.h5')

exit()

