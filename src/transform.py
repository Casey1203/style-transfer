# coding: utf-8

import numpy as np
import tensorflow as tf, pdb

def net(image):
	conv1 = _conv_layer(image, 32, 9, 1) # 32： filter number, 9：filter size, 1: stride
	conv2 = _conv_layer(image, 64, 3, 1) #
	conv3 = _conv_layer(image, 128, 9, 1)
	resid1 = _residual_block(conv3, 3)
	resid2 = _residual_block(resid1, 3)
	resid3 = _residual_block(resid2, 3)
	resid4 = _residual_block(resid3, 3)
	resid5 = _residual_block(resid4, 3)
	conv_t1 = _conv_transpose_layer(resid5, 64, 3, 2)
	conv_t2 = _conv_transpose_layer(conv_t1, 64, 3, 2)
	conv_t3 = _conv_layer(conv_t2, 3, 9, 2, relu=False) # 3 filter number: 3 channel

	preds = tf.nn.tanh(conv_t3) * 150 + 255./2


def _conv_layer(net, num_filter, filter_size, strides, relu=True):
	weight_init = _conv_init_vars(net, num_filter, filter_size) # 初始化权重
	strides_shape = [1, strides, strides, 1] # 1: batch, 1:channel
	net = tf.nn.conv2d(net, weight_init, strides_shape, padding='SAME')
	net = _instance_norm(net) # batch normalization
	if relu:
		net = tf.nn.relu(net)
	return net()


def _conv_transpose_layer(net, num_filter, filter_size, strides):
	weight_init = _conv_init_vars(net, num_filter, filter_size, transpose=True) # 初始化权重
	batch_size, rows, cols, in_channels = [i.value() for i in net.get_shape()]
	new_rows, new_cols = int(rows * strides), int(cols*strides)
	new_shape = [batch_size, new_rows, new_cols, num_filter]
	tf_shape = np.stack(new_shape)
	strides_shape = [1, strides, strides, 1] # 1: batch, 1:channel
	net = tf.nn.conv2d_transpose(net, weight_init, tf_shape, strides_shape, padding='SAME')
	return tf.nn.relu(net)


def _residual_block(net, filter_size=3):
	tmp = _conv_layer(net, 128, filter_size, 1)
	return net + _conv_layer(tmp, 128, filter_size, 1, relu=False) # 自身 + 经过两层conv


def _instance_norm(net):
	# tf直接有batch normalized模块
	batch_size, rows, cols, channels = [i.value() for i in net.get_shape()]
	var_shape = channels
	mu, sigma_sq = tf.nn.moments(net, axes=[1, 2], keep_dims=True)
	shift = tf.Variable(tf.zeros(var_shape))
	scale = tf.Variable(tf.ones(var_shape))
	epsilon = 1e-3
	normalized = (net - mu) / (sigma_sq + epsilon) ** 0.5
	return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):
	_, rows, cols, in_channels = [i.value() for i in net.get_shape()]
	if not transpose:
		weights_shape = [filter_size, filter_size, in_channels, out_channels]
	else:
		weights_shape = [filter_size, filter_size, out_channels, in_channels] # 调换in out顺序
	weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1, seed=1), dtype=tf.float32)

