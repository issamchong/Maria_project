import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 357,90 #define image size here width & height 
input_depth = 1 #1: gray image
train_data_dir = '/home/issam/ML/Maria_pproject/dataset/training' #data training path
testing_data_dir = '/home/issam/ML/Maria_pproject/dataset/testing' #data testing path
epochs = 10 #number of training epoch
batch_size =5 #training batch size
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
NUMB_FILTER_L1 = 30
NUMB_FILTER_L2 = 30
NUMB_FILTER_L3 = 30
NUMB_NODE_FC_LAYER = 25

#define input image order shape
if K.image_data_format() == 'channels_first':
    input_shape_val = (input_depth, img_width, img_height)
else:
    input_shape_val = (img_width, img_height, input_depth)

#define the network
model = Sequential()
# Layer 1
model.add(Conv2D(NUMB_FILTER_L1, (5, 5), 
                 input_shape=input_shape_val, 
                 padding='same', name='input_tensor'))
model.add(Activation('relu'))
model.add(MaxPool2D((2, 2)))

# Layer 2
model.add(Conv2D(NUMB_FILTER_L2, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D((2, 2)))

# Layer 3
model.add(Conv2D(NUMB_FILTER_L3, (5, 5), padding='same'))
model.add(Activation('relu'))

# flattening the model for fully connected layer
model.add(Flatten())

# fully connected layer
model.add(Dense(NUMB_NODE_FC_LAYER, activation='relu'))

# output layer
model.add(Dense(train_generator.num_classes, 
                activation='softmax', name='output_tensor'))

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

