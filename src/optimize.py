# coding: utf-8
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

def optimize(
		content_target, style_target, content_weight, style_weight,
		tv_weight, vgg_path, epochs=2, print_iterations=1000,
		batch_size=4, save_path='saver/fns.ckpt', slow=False,
		learning_rate=1e-3, debug=False
):
	mod = len(content_target) * batch_size
	if mod > 0:
		content_target = content_target[:-mod] # 去掉多余的

	style_features = []
	batch_shape = (batch_size, 256, 256, 3) # height, width, channel
	style_shape = (1,) + style_target.shape

	with tf.Graph.as_default(), tf.device('/cpu:0'),tf.Session() as sess:
		style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
		style_image_pro = vgg.preprocess(style_image)
		net = vgg.net(vgg_path, style_image_pro)
		style_pro = np.array(style_target)
