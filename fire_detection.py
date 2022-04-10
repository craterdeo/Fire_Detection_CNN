import tensorflow as tf
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

def detect(str,flag):
    if(flag == True):
        return "Detected " + str
    
    else:
        return "Didn't detect " + str

fire_training = "Dataset_Fire/Training"
training_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip=True, rotation_range=30, height_shift_range=0.2, fill_mode='nearest')

fire_validation = "Dataset_Fire/Validation"
validation_datagen = ImageDataGenerator(rescale = 1./255)

fire_train_generator = training_datagen.flow_from_directory(fire_training,target_size=(224,224),class_mode='categorical',batch_size = 64)

fire_validation_generator = validation_datagen.flow_from_directory(fire_validation, target_size=(224,224),
                                                                     class_mode='categorical', batch_size= 16)

model_fire = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=(224, 224, 3)),
                         tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2)),
tf.keras.layers.Conv2D(256, (5,5), activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2)),
tf.keras.layers.Conv2D(384, (5,5), activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size = (3,3), strides=(2,2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(2048, activation='relu'),
tf.keras.layers.Dropout(0.25),
tf.keras.layers.Dense(1024, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(2, activation='softmax')])
model_fire.compile(loss='categorical_crossentropy',
optimizer=Adam(lr=0.0001),
metrics=['acc'])
history = model_fire.fit(fire_train_generator,steps_per_epoch = 15,epochs = 50,validation_data = fire_validation_generator,
                            validation_steps = 15)

import numpy as np
import os
from matplotlib import pyplot as plt
path = 'Dataset_fire/Testing'
for i in os.listdir(path):
    img = image.load_img(path + '/' + i, target_size = (224,224))
    plt.imshow(img)
    plt.show()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255
    classes = model_fire.predict(x)
    fire_flag = detect("Fire",np.argmax(classes[0]) == 0 )
    print(fire_flag)