# coding: utf-8

import numpy as np
import tensorflow as tf, pdb
# style transfer中的transform网络，是resnet

def net(image):
	conv1 = _conv_layer(image, 32, 9, 1) # 32： filter number, 9：filter size, 1: stride
	conv2 = _conv_layer(conv1, 64, 3, 1) #
	conv3 = _conv_layer(conv2, 128, 9, 1)
	resid1 = _residual_block(conv3, 3)
	resid2 = _residual_block(resid1, 3)
	resid3 = _residual_block(resid2, 3)
	resid4 = _residual_block(resid3, 3)
	resid5 = _residual_block(resid4, 3)
	conv_t1 = _conv_transpose_layer(resid5, 64, 3, 2)
	conv_t2 = _conv_transpose_layer(conv_t1, 64, 3, 2)
	conv_t3 = _conv_layer(conv_t2, 3, 9, 2, relu=False) # 3 filter number: 3 channel

	preds = tf.nn.tanh(conv_t3) * 150 + 255./2
	return preds


def _conv_layer(image, num_filter, filter_size, strides, relu=True):
	weight_init = _conv_init_vars(image, num_filter, filter_size) # 初始化权重
	strides_shape = [1, strides, strides, 1] # 1: batch, 1:channel
	feat_map = tf.nn.conv2d(image, weight_init, strides_shape, padding='SAME')
	feat_map = _instance_norm(feat_map) # batch normalization
	if relu:
		feat_map = tf.nn.relu(feat_map)
	return feat_map


def _conv_transpose_layer(image, num_filter, filter_size, strides):
	weight_init = _conv_init_vars(image, num_filter, filter_size, transpose=True) # 初始化权重
	batch_size, rows, cols, in_channels = [i.value() for i in image.get_shape()]
	new_rows, new_cols = int(rows * strides), int(cols * strides) # 经过反卷积
	new_shape = [batch_size, new_rows, new_cols, num_filter]
	tf_shape = np.stack(new_shape)
	strides_shape = [1, strides, strides, 1] # 1: batch, 1:channel
	image = tf.nn.conv2d_transpose(image, weight_init, tf_shape, strides_shape, padding='SAME')
	return tf.nn.relu(image)


def _residual_block(image, filter_size=3):
	tmp = _conv_layer(image, 128, filter_size, 1)
	return image + _conv_layer(tmp, 128, filter_size, 1, relu=False) # 自身 + 经过两层conv


def _instance_norm(image):
	# tf直接有batch normalized模块
	batch_size, rows, cols, channels = [i.value() for i in image.get_shape()]
	var_shape = channels # 均值和标准差是向量，维度是channel的个数
	mu, sigma_sqrt = tf.nn.moments(image, axes=[1, 2], keep_dims=True)
	shift = tf.Variable(tf.zeros(var_shape))
	scale = tf.Variable(tf.ones(var_shape))
	epsilon = 1e-3
	normalized = (image - mu) / (sigma_sqrt + epsilon) ** 0.5
	return scale * normalized + shift


def _conv_init_vars(image, out_channels, filter_size, transpose=False):
	# 权重初始化
	_, rows, cols, in_channels = [i.value() for i in image.get_shape()]
	if not transpose:
		weights_shape = [filter_size, filter_size, in_channels, out_channels]
	else:
		weights_shape = [filter_size, filter_size, out_channels, in_channels] # 调换in out顺序
	weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1, seed=1), dtype=tf.float32) # seed=1 每次结果一样
	return weights_init

