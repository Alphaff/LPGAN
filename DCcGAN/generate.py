# Use a trained DenseFuse Net to generate fused images

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave
from datetime import datetime
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from Generator import Generator
from Generator import Decoder
from Discriminator import Discriminator1, Discriminator2
import time


def generate(ir_path, vis_path,model_path, index, output_path = None,train=False):
	ir_img = imread(ir_path) / 255.0
	vis_img = imread(vis_path) / 255.0
	#ir_trans_img = imread(ir_trans_path) / 255.0
	#vis_trans_img = imread(vis_trans_path) / 255.0
	ir_dimension = list(ir_img.shape)
	vis_dimension = list(vis_img.shape)
	#ir_trans_dimension = list(ir_img.shape)
	#vis_trans_dimension = list(vis_img.shape)
	ir_dimension.insert(0, 1)
	ir_dimension.append(1)
	vis_dimension.insert(0, 1)
	vis_dimension.append(1)
	#ir_trans_dimension.insert(0, 1)
	#ir_trans_dimension.append(1)
	ir_img = ir_img.reshape(ir_dimension)
	vis_img = vis_img.reshape(vis_dimension)
	#ir_trans_img = ir_trans_img.reshape(ir_dimension)
	#vis_trans_img = vis_trans_img.reshape(vis_dimension)

	with tf.Graph().as_default(), tf.Session() as sess:
		SOURCE_VIS = tf.placeholder(tf.float32, shape = vis_dimension, name = 'SOURCE_VIS')
		SOURCE_IR = tf.placeholder(tf.float32, shape = ir_dimension, name = 'SOURCE_ir')
		# source_field = tf.placeholder(tf.float32, shape = source_shape, name = 'source_imgs')

		# D1 = Discriminator1('Discriminator1')
		# D2 = Discriminator2('Discriminator2')
		G = Generator('Generator')
		D = Decoder('Decoder')
		# generated_img = G.transform(vis = SOURCE_VIS, ir = SOURCE_IR)
		generated_img1 = G.transform1(vis=SOURCE_VIS, ir=SOURCE_IR,ir_trans=(1-SOURCE_IR),
									  vis_trans=1-SOURCE_VIS,train=train, output_path=output_path)
		#print('shape of generated_img1:', generated_img1.shape)
		generated_img2 = G.transform2(vis=SOURCE_VIS, ir=SOURCE_IR,ir_trans=(1-SOURCE_IR),
									  vis_trans=1-SOURCE_VIS,train=train, output_path=output_path)
		#print('shape of generated_img2:', generated_img2.shape)

		#generated_img3 = generated_img1 + generated_img2
		generated_img3 = tf.concat([generated_img1, generated_img2], -1)

		#print('shape of generated_img3:', generated_img3.shape)

		output_image = D.decode(generated_img3)
		print('shape of generated img:',output_image.shape)
		#print('sadfasdf',output_image[0,:,:,0])

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)
		output = sess.run(output_image, feed_dict = {SOURCE_VIS: vis_img, SOURCE_IR: ir_img})
		#print('asfdsf',output[0,:,:,0])
		output = output[0, :, :, 0]
		#print('first pix of output:',output[0,0])
		imsave(output_path + str(index) + '.bmp', output)


# print('generated image shape:', output_image.shape)

# imsave(output_path + str(index) + '/' + str(model_num) + '_ir_us.bmp', IR[0, :, :, 0])
# imsave(output_path + str(index) + '/' + str(model_num) + '_vis_de.bmp', vis_de[0, :, :, 0])


def save_images(paths, datas, save_path, prefix = None, suffix = None):
	if isinstance(paths, str):
		paths = [paths]

	assert (len(paths) == len(datas))

	if not exists(save_path):
		mkdir(save_path)

	if prefix is None:
		prefix = ''
	if suffix is None:
		suffix = ''

	for i, path in enumerate(paths):
		data = datas[i]
		# print('data ==>>\n', data)
		if data.shape[2] == 1:
			data = data.reshape([data.shape[0], data.shape[1]])
		# print('data reshape==>>\n', data)

		name, ext = splitext(path)
		name = name.split(sep)[-1]

		path = join(save_path, prefix + suffix + ext)
		print('data path==>>', path)
		imsave(path, data)
