from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 1)
num_classes = 2

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/letters and nonletters/train',
        target_size=(img_rows, img_cols),
		color_mode='grayscale',
        batch_size=15,
        class_mode='categorical',
		shuffle='True')

validation_generator = val_datagen.flow_from_directory(
        'data/letters and nonletters/validation',
        target_size=(img_rows, img_cols),
		color_mode='grayscale',
        batch_size=15,
        class_mode='categorical',
		shuffle='True')


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=40,
        epochs=10,
		verbose=1,
        validation_data=validation_generator,
        validation_steps=20)

# serialize model to JSON
model_json = model.to_json()
with open("modelzasegmentaciju.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modezasegmentaciju.h5")
print("Saved model to disk")
