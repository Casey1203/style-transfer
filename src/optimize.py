# coding: utf-8
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img
import functools

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
# DEVICES = 'CUDA_VISIBLE_DEVICES'

def _tensor_size(tensor):
	from operator import mul
	return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def optimize(
		content_target, style_target, content_weight, style_weight,
		tv_weight, vgg_path, epochs=2, print_iterations=1000,
		batch_size=4, save_path='saver/fns.ckpt', slow=False,
		learning_rate=1e-3, debug=False
):
	mod = len(content_target) * batch_size
	if mod > 0:
		content_target = content_target[:-mod] # 去掉多余的

	style_features = {}
	batch_shape = (batch_size, 256, 256, 3) # height, width, channel
	style_shape = (1,) + style_target.shape

	with tf.Graph.as_default(), tf.device('/cpu:0'),tf.Session() as sess:
		style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
		# 把style_image直接经过vgg提取特征
		style_image_pro = vgg.preprocess(style_image)
		style_net = vgg.net(vgg_path, style_image_pro)
		style_pro = np.array(style_target) # 转换格式
		for layer in STYLE_LAYERS:
			features = style_net[layer].eval(feed_dict = {style_image: style_pro})
			features = np.reshape(features, (-1, features.shape[3])) # -1: bhwc,
			gamma = np.matmul(features.T, features) # covariance，特征图与特征图之间的关系
			style_features[layer] = gamma
	with tf.Graph.as_default(), tf.Session() as sess:
		x_content = tf.placeholder(tf.float32, shape=batch_shape, name='x_content')
		# 把content图片直接通过vggnet提取特征
		x_pro = vgg.preprocess(x_content)
		content_features = {}
		content_net = vgg.net(vgg_path, x_pro)
		content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]
		# 把content图片先通过transform网络，把style融合，得到和原图片size一样的图片
		preds = transform.net(x_content / 255.0) # /255.0:归一化。把图片先feedforward到transform网络中
		preds_pro = vgg.preprocess(preds)
		net = vgg.net(vgg_path, preds_pro)
		content_size = _tensor_size(content_features[CONTENT_LAYER]) * batch_size
		# 经过生成网络，不经过生成网络，直接在vgg提取特征的图片。两张图片求差异
		content_loss = \
			content_weight * (2 * tf.nn.l2_loss(net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size)
		# 除以content_size，loss与content_size无关

		style_losses = []

		for style_layer in STYLE_LAYERS:
			layer = net[style_layer] # transform网络
			bs, height, width, filters = map(lambda i: i.value, layer.get_shape())
			size = height * width * filters
			feats = tf.reshape(layer, (bs, height * width, filters))
			feats_T = tf.transpose(feats, perm=[0, 2, 1]) # 把filter的维度提前
			gamma = tf.matmul(feats_T, feats) / size # transform网络在风格上的loss，loss与size无关

			style_gamma = style_features[style_layer] # 之前计算的，用vgg直接跑，不用transform网络。当前layer的gamma值
			style_losses.append(2 * tf.nn.l2_loss(gamma - style_gamma) / style_gamma.size)
		style_loss = style_weight * functools.reduce(tf.add, style_losses)

		# total variation denoising
		tv_y_size = _tensor_size(preds[:, 1:, :, :])
		tv_x_size = _tensor_size(preds[:, :, 1:, :])
		y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :batch_shape[1]-1, :, :])
		x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :batch_shape[2]-1, :])
		tv_loss = tv_weight * 2 * (x_tv/tv_x_size + y_tv/tv_y_size) / batch_size

		loss = content_loss + style_loss + tv_loss

		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		sess.run(tf.global_variables_initializer())
		for epoch in xrange(epochs):
			num_examples = len(content_target)
			iterations = 0
			while iterations * batch_size < num_examples:
				start_time = time.time()
				curr = iterations * batch_size
				step = curr + batch_size
				x_batch = np.zeros(batch_shape, dtype=np.float32)
				for j, img_p in enumerate(content_target[curr: step]):
					x_batch[j] = get_img(img_p, (256, 256, 3)).astype(np.float32)
				iterations += 1
				assert x_batch.shape[0] == batch_size

				feed_dict = {x_content: x_batch}

				train_step.run(feed_dict)
				end_time = time.time()
				delta_time = end_time - start_time

				is_print_iter = int(iterations) % print_iterations == 0
				is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples

				should_print = is_print_iter or is_last

				if should_print:
					to_get = [style_loss, content_loss, loss, preds]
					test_dict = {x_content: x_batch}
					tmp = sess.run(to_get, feed_dict=test_dict)
					_style_loss, _content_loss, _loss, _preds = tmp
					losses = (_style_loss, _content_loss, _loss)

					saver = tf.train.Saver()

					res = saver.save(sess, save_path)
					yield (_preds, losses, iterations, epoch)


