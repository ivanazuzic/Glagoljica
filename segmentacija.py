import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import shutil

cnt = 0

def prilagodi(img):
    BLUE = [255,0,0]
    imgsize = img.shape
    w = imgsize[0]
    h = imgsize[1]
    gore = 0
    dolje = 0
    lijevo = 0
    desno = 0
    if h > w:
        diff = h - w
        lijevo = int(diff / 2) + (diff % 2)
        desno = int(diff / 2)
    else:
        diff = w - h
        gore = int(diff / 2) + (diff % 2)
        dolje = int(diff / 2)
    img = cv2.copyMakeBorder(img, lijevo, desno, gore, dolje, cv2.BORDER_CONSTANT, value=BLUE)
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    return img

def segmentacija_slova(img2, h1, h2):
    global cnt
    histogram2 = []
    height2, width2 = img2.shape
    maxfill = 0
    for j in range(width2):
        sum = 0
        for i in range(height2):
            sum += img2[i][j]
        histogram2.append(sum)  
        maxfill = max(maxfill, sum)

    for i in range(width2):
        histogram2[i] = histogram2[i] / maxfill

    plt.plot(range(len(histogram2)), histogram2, 'ro')
    plt.show()

    top2 = 0
    inpicture2 = 0
    for i in range(1, width2):
        if histogram2[i] > 0.99:
            if inpicture2 == 1:
                """
                cv2.imshow("Redak", img2[:, top2:i])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                """
                cnt += 1
                cv2.imwrite("Slike/" + str(cnt) + ".png", prilagodi(img2[:, top2:i]))
            top2 = i
            inpicture2 = 0
        else:
            inpicture2 = 1


def segmentacija_redaka(img):
    global cnt
    histogram = []
    height, width = img.shape
    maxfill = 0
    for i in range(height):
        sum = 0
        for j in range(width):
            sum += img[i][j]
        histogram.append(sum)   
        maxfill = max(maxfill, sum)
    
    for i in range(height):
        histogram[i] = histogram[i] / maxfill

    plt.plot(range(len(histogram)), histogram, 'ro')
    plt.show()

    top = 0
    inpicture = 0
    for i in range(1, height):
        #print(histogram[i])
        if histogram[i] > 0.99: #boja
            if inpicture == 1:
                segmentacija_slova(img[top:i, :], top, i)
                """
                cv2.imshow("Redak", img[top:i, :])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                """
            top = i
            inpicture = 0
        else:                   #nema boje
            inpicture = 1
    if inpicture == 1:
        segmentacija_slova(img[top:i, :], top, i)
        cv2.imshow("Redak", img[top:i, :])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# load json and create model
json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model2.h5")
print("Loaded model from disk")

original = cv2.imread('azbuka olovka.png',0)
#original = cv2.imread('slika.png',0)
ret, img = cv2.threshold(original,127,255,cv2.THRESH_BINARY)

segmentacija_redaka(img)

# evaluate loaded model on test data
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
              
img_rows, img_cols = 100, 100
num_classes = 33

X = []

path = "Slike/"
imlist = os.listdir(path)
#print(imlist)
imarray = [np.array(Image.open(path + im)).flatten() for im in imlist]
#print(imarray)
imarray = shuffle(imarray, random_state=2)

X += imarray

X = np.array(X)

train_samples = len(X)

if K.image_data_format() == 'channels_first':
    X = X.reshape(train_samples, 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X = X.reshape(train_samples, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X = X.astype('float32')

X /= 255

output = loaded_model.predict(X, verbose=1)

azbuka = ['a', 'b', 'v', 'g', 'd', 'e', 'zj', 'dz', 'z', '(i)', 'i', 'dj', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'f', 'h', '(o)', "(sj)c'", 'c', 'cj', 'sj', 'ja, (i)je', 'ju' ,'j', 'poluglas']
azbuka.sort()

for i in range(len(output)):
    indeks = output[i].argmax()
    print(indeks, azbuka[indeks], output[i])