from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt

img_rows, img_cols = 50, 50
input_shape = (img_rows, img_cols, 1)
num_classes = 33

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/letters/train',
        target_size=(img_rows, img_cols),
		color_mode='grayscale',
        batch_size=120,
        class_mode='categorical',
		shuffle='True')

validation_generator = val_datagen.flow_from_directory(
        'data/letters/validation',
        target_size=(img_rows, img_cols),
		color_mode='grayscale',
        batch_size=120,
        class_mode='categorical',
		shuffle='True')


model = Sequential()
    #C1 Layer
    model.add(Convolution2D(32, filter_size, filter_size, border_mode='same', input_shape=(28,28,1)))
    # The activation for layers is ReLU
    model.add(Activation('relu'))
    # Max pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #Second Layer
    model.add(Convolution2D(num_filters, filter_size, filter_size, border_mode='valid', input_shape=(14,14,1)))
    model.add(Activation('relu'))
    # Max pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #Third Layer
    model.add(Convolution2D(num_filters, filter_size, filter_size, border_mode='valid', input_shape=(5,5,1)))
    #Flatten the CNN output
    model.add(Flatten())
    #Add Three dense Layer for the FNN 
    
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(32))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(10))
    # For classification, the activation is softmax
    model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit_generator(
        train_generator,
        steps_per_epoch=120,
        epochs=12,
		verbose=1,
        validation_data=validation_generator,
        validation_steps=120)

# serialize model to JSON
model_json = model.to_json()
with open("modelmreze2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelmreze2.h5")
print("Saved model to disk")


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

