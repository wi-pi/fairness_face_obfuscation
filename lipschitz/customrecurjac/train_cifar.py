import keras
import numpy as np
import keras.backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import glorot_normal, RandomNormal
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from setup_cifar import CIFAR
from keras.callbacks import ModelCheckpoint


BALANCED = False

if BALANCED:
    balanced_str = 'balanced'
else:
    balanced_str = 'imbalanced'

data = CIFAR()
if BALANCED:
    x_train = data.balanced_data
    y_train = data.balanced_labels
else:
    x_train = data.imbalanced_data
    y_train = data.imbalanced_labels

# x_train = data.train_data
# y_train = data.train_labels

x_test = data.validation_data
y_test = data.validation_labels

num_classes = 10
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
    )
datagen.fit(x_train)


def create_model(s = 2, weight_decay = 1e-2, act="relu"):

    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3,3), padding='same', use_bias=True, kernel_initializer=glorot_normal(), input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 2
    model.add(Conv2D(128, (3,3), padding='same', use_bias=True, kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 3
    model.add(Conv2D(128, (3,3), padding='same', use_bias=True, kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 4
    model.add(Conv2D(128, (3,3), padding='same', use_bias=True, kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    # First Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))
    
    
    # Block 5
    model.add(Conv2D(128, (3,3), padding='same', use_bias=True, kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 6
    model.add(Conv2D(128, (3,3), padding='same', use_bias=True, kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 7
    model.add(Conv2D(256, (3,3), padding='same', use_bias=True, kernel_initializer=glorot_normal()))
    # Second Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    
    # Block 8
    model.add(Conv2D(256, (3,3), padding='same', use_bias=True, kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 9
    model.add(Conv2D(256, (3,3), padding='same', use_bias=True, kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    # Third Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    
    
    # Block 10
    model.add(Conv2D(512, (3,3), padding='same', use_bias=True, kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))

    # Block 11  
    model.add(Conv2D(2048, (1,1), padding='same', kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 12  
    model.add(Conv2D(256, (1,1), padding='same', kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    # Fourth Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))


    # Block 13
    model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    # Fifth Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))

    # Final Classifier
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return model


if __name__ == '__main__':
    # Prepare for training 
    model = create_model(act="relu")
    batch_size = 128
    epochs = 25
    train = {}

    # First training for 50 epochs - (0-50)
    opt_adm = keras.optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_1"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                        steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
                                        verbose=1,validation_data=(x_test,y_test))
    model.save("models/simplenet_{}_generic_first.h5".format(balanced_str))
    print(train["part_1"].history)

    # Training for 25 epochs more - (50-75)
    opt_adm = keras.optimizers.Adadelta(lr=0.7, rho=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_2"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                        steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
                                        verbose=1,validation_data=(x_test,y_test))
    model.save("models/simplenet_{}_generic_second.h5".format(balanced_str))
    print(train["part_2"].history)

    # Training for 25 epochs more - (75-100)
    opt_adm = keras.optimizers.Adadelta(lr=0.5, rho=0.85)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_3"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                        steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
                                        verbose=1,validation_data=(x_test,y_test))
    model.save("models/simplenet_{}_generic_third.h5".format(balanced_str))
    print(train["part_3"].history)

    # Training for 25 epochs more  - (100-125)
    opt_adm = keras.optimizers.Adadelta(lr=0.3, rho=0.75)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_4"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                        steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
                                        verbose=1,validation_data=(x_test,y_test))
    model.save("models/simplenet_{}_generic_fourth.h5".format(balanced_str))
    print(train["part_4"].history)
    print("\n \n Final Logs: ", train)
