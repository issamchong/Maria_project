import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

img_width, img_height = 357,90 #define image size here width & height 
input_depth = 1 #1: gray image
train_data_dir = '/home/issam/ML/Maria_project/dataset/training' #data training path
testing_data_dir = '/home/issam/ML/Maria_project/dataset/testing' #data testing path
epochs = 3 #number of training epoch
batch_size = 6 #training batch size
#train_datagen = ImageDataGenerator(
       #rescale=1/255,
       #rotation_range=0.5,
        #width_shift_range=0.05,
        #height_shift_range=0.05,
        #shear_range=0.05,
        #zoom_range=0.05,
        #horizontal_flip=True,
        #fill_mode='nearest')
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='categorical')

testing_generator = test_datagen.flow_from_directory(
    testing_data_dir,
    color_mode='grayscale',
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='categorical')


# define number of filters and nodes in the fully connected layer
NUMB_FILTER_L1 = 32
NUMB_FILTER_L2 = 64
NUMB_FILTER_L3 = 80
NUMB_FILTER_L4 = 100

NUMB_NODE_FC_LAYER = 70

#define input image order shape
if K.image_data_format() == 'channels_first':
    input_shape_val = (input_depth, img_width, img_height)
else:
    input_shape_val = (img_width, img_height, input_depth)

#define the network
model = Sequential()
# Layer 1
model.add(Conv2D(NUMB_FILTER_L1, kernel_size=(3, 3), 
                 activation='relu',
                 input_shape=input_shape_val, 
                 padding='same', name='input_tensor'))


# Layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(Dense(9, activation='softmax'))


# Compilile the network
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])
# Show the model summary
model.summary()

# Train and test the network
model.fit_generator(
    train_generator,#our training generator
    #number of iteration per epoch = number of data / batch size
    steps_per_epoch=np.floor(train_generator.n/batch_size),
    epochs=epochs,#number of epoch
    validation_data=testing_generator,#our validation generator
    #number of iteration per epoch = number of data / batch size
    validation_steps=np.floor(testing_generator.n / batch_size))
print("Training is done!")
model.save('/home/issam/ML/Maria_model.h5')
print("Model is successfully stored!")
print("all good")


#class_names=

