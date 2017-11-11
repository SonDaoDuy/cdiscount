from __future__ import print_function
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import segnet

batch_size = 1
nb_classes = 2
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 256, 256
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

model = segnet.build_segnet(input_shape=(img_channels, img_rows, img_cols))


if not data_augmentation:
	print("Not using data argumentation.")
	model.fit(X_train, Y_train,
			batch_size=batch_size,
			nb_epoch=nb_epoch,
			validation_data=(X_test, Y_test),
			shuffle=True)
else:
	print('Using real-time data argumentation.')
	# This will do preprocessing and realtime data augmentation:
	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,  # randomly flip images
		vertical_flip=False)  # randomly flip images

	# Compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(X_train)
	model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
						steps_per_epoch=X_train.shape[0] // batch_size,
						validation_data=(X_test, Y_test),
						epochs=nb_epoch, verbose=1, max_q_size=100)