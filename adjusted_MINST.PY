from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os 




batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

beds_train = os.listdir('/home/issam/ML/Maria_project/dataset/training/bed') 
beds_test = os.listdir('/home/issam/ML/Maria_project/dataset/testing/bed') 

desks_train = os.listdir('/home/issam/ML/Maria_project/dataset/training/desk') 
desks_test = os.listdir('/home/issam/ML/Maria_project/dataset/testing/desk') 

bed_train=np.empty([357,90],dtype='uint8',order='C')
bed_test=np.empty([357,90],dtype='uint8',order='C')

desk_train=np.empty([357,90],dtype='uint8',order='C')
desk_test=np.empty([357,90],dtype='uint8',order='C')


for f in beds_test:
    img=np.asarray(f)
    np.append(bed_test,img)

for f in beds_train:
    img=np.asarray(f)
    np.append(bed_train,img)

for f in beds_test:
    img=np.asarray(f)
    np.append(bed_test,img)

for f in desk_train:
    img=np.asarray(f)
    np.append(desk_train,img)

for f in desk_test:
    img=np.asarray(f)
    np.append(desk_test,img)




# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_test)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
  
