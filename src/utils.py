import scipy.misc, numpy as np, os, sys


def get_img(src):
	img = scipy.misc.imread(src, mode='RGB')
	return img

def save_img(img, src):
	scipy.misc.imsave(src, img)
