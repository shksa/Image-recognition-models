import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from cs231n.data_utils import get_CIFAR10_data
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras import initializations


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# Get the data from the dataset folder
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# from cs231 script
# data = get_CIFAR10_data()
# X_train, y_train = data["X_train"], data["y_train"]
# X_val, y_val = data["X_val"], data["y_val"]
# X_test, y_test = data["X_test"], data["y_test"]


# Assert the datatype to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# X_val = X_val.astype('float32')

N, C, H, W = X_train.shape

# Train/Val split
# mean_image = np.mean(X_train, axis=0)
mean_of_channels = ((1. / (N * H * W)) * np.sum(X_train, axis=(0, 2, 3))).reshape(1, C, 1, 1)
var_of_channels = (1. / (N * H * W) * np.sum((X_train)**2,axis=(0, 2, 3))).reshape(1, C, 1, 1)
X_val = X_train[np.arange(49000, 50000)]
y_val = y_train[np.arange(49000, 50000)]
X_train = X_train[np.arange(49000)]
y_train = y_train[np.arange(49000)]


# Pre-processing the data:

# Zero-center the data: subtract the mean image
# X_train -= mean_image
# X_val -= mean_image
# X_test -= mean_image


# # Normalize input dimensions from above (-140.269 - 155.005) to a smaller scale (-0.550074 - 0.607863)
# X_train = X_train / 255
# X_val = X_val / 255
# X_test = X_test / 255


# Zero-mean and unit variance the data: subtract the mean stats and divide by std deviation
X_train = (X_train - mean_of_channels) / np.sqrt(var_of_channels + 1e-8)
X_val = (X_val - mean_of_channels) / np.sqrt(var_of_channels + 1e-8)
X_test = (X_test - mean_of_channels) / np.sqrt(var_of_channels + 1e-8)


# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)
# print("y_train: ", y_train.shape, "value-type: ", type(y_train[0, 0]))
# print("y_val: ", y_val.shape, "value-type: ", type(y_val[0, 0]))
# print("y_test: ", y_test.shape, "value-type: ", type(y_test[0, 0]))


# Hyperparameters
num_classes = 10
l_rate = 0.01 
reg = 0.0001 #// Deprecated, use maxnorm weight-constraint instead as a regularization technique
# weight_scale = 0.001 // Deprecated, use he-normal or glorot normal
hidden_dims = (1024, 512)
num_filters = (32, 64, 128)
filter_size = 3
epochs = 25
b_size = 64
decay_rate = l_rate / epochs


default_of_keras = 'glorot_normal'
my_init = 'he_normal'

# Model architecture
model = Sequential()

# Batch Normalization on input training batch
# model.add(BatchNormalization(axis=1, input_shape=(3, 32, 32)))

# BLOCK - 1
model.add(Convolution2D(num_filters[0], filter_size, filter_size, init=my_init, border_mode='same', W_constraint=maxnorm(4), input_shape=(3, 32, 32)))
model.add(Activation('relu'))
#model.add(BatchNormalization(axis=1))
model.add(Dropout(0.2))
model.add(Convolution2D(num_filters[0], filter_size, filter_size, init=my_init, border_mode='same', W_constraint=maxnorm(4)))
model.add(Activation('relu'))
# model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization(axis=1))
# BLOCK - 2
model.add(Convolution2D(num_filters[1], filter_size, filter_size, init=my_init, border_mode='same', W_constraint=maxnorm(4)))
model.add(Activation('relu'))
#model.add(BatchNormalization(axis=1))
model.add(Dropout(0.2))
model.add(Convolution2D(num_filters[1], filter_size, filter_size, init=my_init, border_mode='same', W_constraint=maxnorm(4)))
model.add(Activation('relu'))
# model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization(axis=1))
# BLOCK - 3
model.add(Convolution2D(num_filters[2], filter_size, filter_size, init=my_init, border_mode='same', W_constraint=maxnorm(4)))
model.add(Activation('relu'))
#model.add(BatchNormalization(axis=1))
model.add(Dropout(0.2))
model.add(Convolution2D(num_filters[2], filter_size, filter_size, init=my_init, border_mode='same', W_constraint=maxnorm(4)))
model.add(Activation('relu'))
# model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization(axis=1))
# PSUEDO BLOCK
model.add(Flatten())
# BLOCK - 4
model.add(Dropout(0.2))
model.add(Dense(hidden_dims[0], init=my_init, W_constraint=maxnorm(4)))
model.add(Activation('relu'))
#model.add(BatchNormalization(mode=1))
# BLOCK - 5
model.add(Dropout(0.5))
model.add(Dense(hidden_dims[1], init=my_init, W_constraint=maxnorm(4)))
model.add(Activation('relu'))
#model.add(BatchNormalization(mode=1))
# BLOCK - 6
model.add(Dropout(0.5))
model.add(Dense(num_classes, init=my_init))
model.add(Activation('softmax'))


# Compile the model
# Best choices: 1)adam, 2)sgd+nestorov, 3)Adadelta
sgd = SGD(lr=l_rate, decay=decay_rate, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the model GOAL:- Traing acc: 84% Got more than that
training_stats = model.fit(X_train, y_train,
                           nb_epoch=epochs, batch_size=b_size, verbose = 1,
                           validation_data=(X_val, y_val))


# Evaluation on Validation set
scores = model.evaluate(X_val, y_val, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# Evaluation on Test set
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# Plotting training time loss over all the epochs
plt.subplot(2, 1, 1)
plt.plot(training_stats.history['loss'], 'o')
plt.xlabel('epoch')
plt.ylabel('loss')

# Plotting training accuracy, validation accuray over all the epochs
plt.subplot(2, 1, 2)
plt.plot(training_stats.history['acc'], '-o')
plt.plot(training_stats.history['val_acc'], '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
