from __future__ import division
from keras.models import Model
from keras.layers.convolutional import (
	Conv2D, 
	MaxPooling2D, 
	Conv2DTranspose)
from keras.layers.normalization import BatchNormalization
from keras.layers import (
	Input, 
	Activation, 
	Concatenate)
from keras import backend as K

#change all axis = 3 if  K.image_data_format() = 'channel_last'

def bn_conv_relu(input_tensor, kernel_size, filters, momentum):
	"""
	batch norm convolutional relu
	"""
	x = BatchNormalization(axis=1, momentum=momentum)(input_tensor)
	x = Conv2D(filters, kernel_size, padding='same')(x)
	x = Activation('relu')(x)
	return x

def bn_upconv_relu(input_tensor, kernel_size, filters, momentum):
	"""
	batch norm upconvolutional relu
	can use UpSampling2D instead Conv2DTranspose
	"""
	x = BatchNormalization(axis=1, momentum=momentum)(input_tensor)
	x = Conv2DTranspose(filters, kernel_size, strides=(2,2), activation='relu', padding='same')(x)
	return x

def downward_block(no_of_conv, input_tensor, kernel_size, filters):
	""" An downward block
	# Arguments
		no_of_conv: number of bn_conv_layer 2 or 3
		input_tensor: channel*height*width
		kernel_size: (a,b)
		filters: b
	# Return
		[downsample_tensor, concat_tensor]

	"""
	momentum = 0.01
	if no_of_conv == 2:
		y = bn_conv_relu(input_tensor, kernel_size, filters, momentum)
		x = bn_conv_relu(y, kernel_size, filters, momentum)
		x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
		return x, y
	if no_of_conv == 3:
		x = bn_conv_relu(input_tensor, kernel_size, filters, momentum)
		y = bn_conv_relu(x, kernel_size, filters, momentum)
		x = bn_conv_relu(y, kernel_size, filters, momentum)
		x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
		return x, y

def upward_block(concat, input_tensor, kernel_size, filters_down, filters_up):
	""" An upward block
	# Arguments
		concat: 0 (no concat) or 1 (concat)
		input_tensor: list of 2 tensors [output_of_last_layer, concat_tensor]
		kernel_size: (a,b)
		filters_down: no of filters downward [a,b]
		filters_up: no of filters upward a
	# Return 
		upsample_tensor
	"""
	if concat == 1:
		x = Concatenate(axis = 1, input_tensor)
		# x = merge([UpSampling2D(size=(2, 2))(input_tensor[0]), input_tensor[1]], mode='concat', concat_axis=1) 
	else:
		x = input_tensor[0]

	momentum = 0.01
	filter1, filter2 = filters_down
	x = bn_conv_relu(x, kernel_size, filter1, momentum)
	x = bn_conv_relu(x, kernel_size, filter2, momentum)
	x = bn_upconv_relu(x, kernel_size, filters_up, momentum)

	return x

def final_block(input_tensor, kernel_size, filters_down):
	""" Final block
	# Arguments
		input_tensor: list of 2 tensors [output_of_last_layer, concat_tensor]
		kernel_size: (a,b)
		filters_down: no of filters downward [a,b]
	# Return 
		output_img
	"""
	x = Concatenate(input_tensor)
	momentum = 0.01
	filter1, filter2 = filters_down
	x = bn_conv_relu(x, kernel_size, filter1, momentum)
	x = bn_conv_relu(x, kernel_size, filter2, momentum)
	x = Conv2D(1, (1,1), strides=(1,1), activation='sigmoid')(x)

	return x


def build_segnet(input_tensor=None, input_shape=None):
	""" Build the segnet
	# Arguments
		input_tensor: tensor of img input 
		input_shape: (no_channel, height, width) shape of input
	# Return
		segnet model
	"""

	# using this line to transfer format of img input to (no_channel, heigh, width)
	K.set_image_data_format('channel_first')

	if input_shape is None:
		input_shape = (1, 256, 256)

	if len(input_shape) != 3:
		raise Exception("Input shape should be a tuple (no_channels, height, width)")
	
	if input_tensor is None:
		img_input = Input(shape=input_shape)
	else:
		if not K.is_keras_tensor(input_tensor):
			img_input = Input(tensor=input_tensor, shape=input_shape)
		else:
			img_input = input_tensor

	x = Conv2D(64, (3,3), padding='same', activation='relu')(img_input)

	x, concat1 = downward_block(2, x, (3,3), 64)
	x, concat2 = downward_block(3, x, (3,3), 64)
	x, concat3 = downward_block(3, x, (3,3), 64)
	x, concat4 = downward_block(3, x, (3,3), 64)
	x, concat5 = downward_block(3, x, (3,3), 64)

	x = upward_block(0, [x], (3,3), [96,64], 64)
	x = upward_block(1, [x, concat1], (3,3), [96,64], 64)
	x = upward_block(1, [x, concat2], (3,3), [96,64], 64)
	x = upward_block(1, [x, concat3], (3,3), [96,64], 64)
	x = upward_block(1, [x, concat4], (3,3), [96,64], 64)

	x = final_block([x, concat5], (3,3), [96,64])

	model = Model(img_input, x, name='segnet')
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	return model



