# coding: utf-8
import tensorflow as tf
import numpy as np
import scipy.io
import pdb

MEAN_PIXEL = [123.68, 116.779, 103.939]



def net(data_path, input_image):
	layers = (
		'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

		'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

		'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
		'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

		'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
		'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

		'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
		'relu5_3', 'conv5_4', 'relu5_4', 'pool5',

		'fc6', 'relu6',

		'fc7', 'relu7',

		'fc8', 'softmax' #''prob'
	)

	data = scipy.io.loadmat(data_path)
	mean = data['normalization'][0][0][0]
	mean_pixel = np.mean(mean, axis=(0, 1))
	weights = data['layers'][0]

	net = {}
	current = input_image
	for i, name in enumerate(layers):
		kind = name[:4]
		if kind == 'conv':
			kernels, bias = weights[0][0][0][0]
			# matconvnet: weights are [width, height, in_channels, out_channels]
			# tensorflow: weights are [height, width, in_channels, out_channels]
			kernels = np.transpose(kernels, (1, 0, 2, 3))
			bias = bias.reshape(-1)
			current = _conv_layer(current, kernels, bias)
		elif kind == 'relu':
			current = tf.nn.relu(current)
		elif kind == 'pool':
			current = _pool_layer(current)
		elif kind == 'soft':
			current = _softmax_preds(current)

		kind2 = name[:2]
		if kind2 == 'fc':
			# print(weights)
			kernels, bias = weights[0][0][0][0]
			kernels = kernels.reshape(-1, kernels.shape[-1])
			bias = bias.reshape(-1)
			current = _fc_layer(current, kernels, bias)

		net[name] = current

	assert len(net) == len(layers)
	return net


def _conv_layer(input, weights, bias):
	conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
			padding='SAME')
	return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
	return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
			padding='SAME')

def _fc_layer(input, weights, bias):
	shape = input.get_shape().as_list()
	dim = 1
	for d in shape[1:]:
		dim *= d
	x = tf.reshape(input, [-1, dim])

	Wx_plus = tf.matmul(x, weights)
	fc = tf.nn.bias_add(Wx_plus, bias)
	return fc

def _softmax_preds(input):
	preds = tf.nn.softmax(input, name="prediction")
	return preds

def preprocess(image):
	# 使用vgg网络要先去中心化
	return image - MEAN_PIXEL


def unprocess(image):
	return image + MEAN_PIXEL