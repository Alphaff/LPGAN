import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
# from Deconv import deconv_vis, deconv_ir
import cv2
from scipy.misc import imread, imsave
#import matplotlib.pyplot as
from PIL import Image

#sess = tf.InteractiveSession()

WEIGHT_INIT_STDDEV = 0.05

class Generator(object):

	def __init__(self, sco):
		self.encoderA = EncoderA(sco)
		self.encoderB = EncoderB(sco)
		self.decoder = Decoder(sco)

	def transform1(self, vis, ir,ir_trans=None,vis_trans=None,train=True, output_path=None):
		# IR = deconv_ir(ir, strides = [1, 4, 4, 1], scope_name = 'deconv_ir')
		# VIS = deconv_vis(vis, strides = [1, 1, 1, 1], scope_name = 'deconv_vis')
		#print('**start trandform1')
		#img = vis

		img = tf.concat([vis,vis,vis,vis,ir],3)
		#img = tf.concat([vis, vis, vis, vis_trans, ir], 3)
		#print('通道数：',img.shape[-1])
		#print('start')
		code = self.encoderA.encode(img, train=train, output_path=output_path)
		#print('end')
		# self.target_features = code
		# generated_img = self.decoder.decode(self.target_features)
		return code

	def transform2(self, vis, ir,ir_trans=None,vis_trans=None, train=True, output_path=None):
		# IR = deconv_ir(ir, strides = [1, 4, 4, 1], scope_name = 'deconv_ir')
		# VIS = deconv_vis(vis, strides = [1, 1, 1, 1], scope_name = 'deconv_vis')
		#print('**start trandform2')
		#img = ir

		img = tf.concat([ir,ir,ir,ir,vis],3)
		#img = tf.concat([ir, ir, ir, ir_trans, vis], 3)
		code = self.encoderB.encode(img, train=train, output_path=output_path)
		# self.target_features = code
		# generated_img = self.decoder.decode(self.target_features)
		return code


class EncoderA(object):
	def __init__(self, scope_name):
		self.scopeA = scope_name
		self.weight_varsA = []
		with tf.variable_scope(self.scopeA):
			with tf.variable_scope(self.scopeA):
				with tf.variable_scope('encoderA'):
					self.weight_varsA.append(self._create_variablesA(5, 16, 3, scope='Aconv1_1'))
					self.weight_varsA.append(self._create_variablesA(16, 16, 3, scope='Adense_block_conv1'))
					self.weight_varsA.append(self._create_variablesA(32, 16, 3, scope='Adense_block_conv2'))
					self.weight_varsA.append(self._create_variablesA(48, 16, 3, scope='Adense_block_conv3'))
					#self.weight_varsA.append(self._create_variablesA(64, 16, 3, scope = 'Adense_block_conv4'))
					#self.weight_varsA.append(self._create_variablesA(80, 16, 3, scope='Adense_block_conv5'))

					'''self.weight_varsA.append(self._create_variablesA(5, 32, 3, scope='Aconv1_1'))
					self.weight_varsA.append(self._create_variablesA(32, 32, 3, scope='Adense_block_conv1'))
					self.weight_varsA.append(self._create_variablesA(64, 32, 3, scope='Adense_block_conv2'))
					self.weight_varsA.append(self._create_variablesA(96, 32, 3, scope='Adense_block_conv3'))'''
	def encode(self, image,train=True, output_path=None):
		dense_indices = [1, 2, 3, 4,5,6]
		out = image
		#pic = np.array()
		for i in range(len(self.weight_varsA)):
			kernel, bias = self.weight_varsA[i]
			if i in dense_indices:
				out = conv2d(out, kernel, bias, dense = True, use_relu = True,
				             Scope = self.scopeA + '/encoderA/b' + str(i))
			else:
				out = conv2d(out, kernel, bias, dense = False, use_relu = True,
				             Scope = self.scopeA + '/encoderA/b' + str(i))
				#print('shape of out:',out[0].shape)
				#print('shape of out:',out.shape)

			#print('encodeA success')
			#print('A第%d层编码'%(i+1))
			if i == len(self.weight_varsA)-1:print('encodeA end')
		return out

	def _create_variablesA(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV),
								 name='kernelA')
			bias = tf.Variable(tf.zeros([output_filters]), name='biasA')
		return (kernel, bias)


class EncoderB(object):
	def __init__(self, scope_name):
		self.scopeB = scope_name
		self.weight_varsB = []
		self.info1 = 0
		self.info2 = 0
		with tf.variable_scope(self.scopeB):
			with tf.variable_scope(self.scopeB):
				with tf.variable_scope('encoderB'):
					self.weight_varsB.append(self._create_variablesB(5, 16, 3, scope='Bconv1_1'))
					self.weight_varsB.append(self._create_variablesB(16, 16, 3, scope='Bdense_block_conv1'))
					self.weight_varsB.append(self._create_variablesB(32, 16, 3, scope='Bdense_block_conv2'))
					self.weight_varsB.append(self._create_variablesB(48, 16, 3, scope='Bdense_block_conv3'))
					#self.weight_varsB.append(self._create_variablesB(64, 16, 3, scope = 'Bdense_block_conv4'))
					#self.weight_varsB.append(self._create_variablesB(80, 16, 3, scope='Bdense_block_conv5'))

					'''self.weight_varsB.append(self._create_variablesB(5, 32, 3, scope='Bconv1_1'))
					self.weight_varsB.append(self._create_variablesB(32, 32, 3, scope='Bdense_block_conv1'))
					self.weight_varsB.append(self._create_variablesB(64, 32, 3, scope='Bdense_block_conv2'))
					self.weight_varsB.append(self._create_variablesB(96, 32, 3, scope='Bdense_block_conv3'))'''

	# self.weight_vars.append(self._create_variables(80, 32, 3, scope = 'dense_block_conv5'))

	# self.weight_vars.append(self._create_variables(96, 16, 3, scope = 'dense_block_conv6'))
	def encode(self, image, train=True, output_path=None):
		dense_indices = [1, 2, 3, 4,5,6]
		out = image

		for i in range(len(self.weight_varsB)):
			kernel, bias = self.weight_varsB[i]
			if i in dense_indices:
				out = conv2d(out, kernel, bias, dense = True, use_relu = True,
				             Scope = self.scopeB + '/encoderB/b' + str(i))
				'''
				for j in range(16):
					self.info1 += tf.square(tf.norm(out[0,:,:,j], axis = [0, 1], ord = 'fro'))
					with tf.Session() as sess:
						#sess.run(tf.global_variables_initializer())
						print('information1:',sess.run(self.info1,feed_dict={out[:,:,:,j],out[:,:,:,j]}))
					#sess
					#print(tf.nn.softmax_cross_entropy_with_logits_v2(self.info1,self.info2))
				'''
			else:
				out = conv2d(out, kernel, bias, dense = False, use_relu = True,
				             Scope = self.scopeB + '/encoderB/b' + str(i))
			'''
			for j in range(16):
					self.info2 += tf.square(tf.norm(out[0,:,:,j], axis = [0, 1], ord = 'fro'))
					with tf.Session() as sess:
						#sess.run(tf.global_variables_initializer())
						print('information2:',sess.run(self.info2,feed_dict={out[:,:,:,j],out[:,:,:,j]}))
					    '''
			#print('encodeB success')
			#print('B第%d层编码'%(i+1))
			if i == len(self.weight_varsB)-1:print('encodeB end')
		return out

	def _create_variablesB(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV),
								 name='kernelB')
			bias = tf.Variable(tf.zeros([output_filters]), name='biasB')
		return (kernel, bias)




class Decoder(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name
		with tf.name_scope(scope_name):
			with tf.variable_scope('decoder'):
				'''self.weight_vars.append(self._create_variables(192, 96, 3, scope='conv2_1'))
				self.weight_vars.append(self._create_variables(96, 48, 3, scope='conv2_2'))
				self.weight_vars.append(self._create_variables(48, 24, 3, scope='conv2_3'))
				self.weight_vars.append(self._create_variables(24, 12, 3, scope='conv2_4'))
				self.weight_vars.append(self._create_variables(12, 1, 3, scope='conv2_5'))'''

				self.weight_vars.append(self._create_variables(128, 64, 3, scope='conv2_1'))
				self.weight_vars.append(self._create_variables(64, 32, 3, scope='conv2_2'))
				self.weight_vars.append(self._create_variables(32, 16, 3, scope='conv2_3'))
				self.weight_vars.append(self._create_variables(16, 1, 3, scope='conv2_4'))

				'''self.weight_vars.append(self._create_variables(256, 128, 3, scope='conv2_1'))
				self.weight_vars.append(self._create_variables(128, 64, 3, scope='conv2_2'))
				self.weight_vars.append(self._create_variables(64, 32, 3, scope='conv2_3'))
				self.weight_vars.append(self._create_variables(32, 16, 3, scope='conv2_4'))
				self.weight_vars.append(self._create_variables(16, 1, 3, scope='conv2_5'))'''
		# self.weight_vars.append(self._create_variables(32, 1, 3, scope = 'conv2_4'))

	# self.weight_vars.append(self._create_variables(16, 1, 3, scope = 'conv2_5'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			#shape = [kernel_size, kernel_size, input_filters, output_filters]
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def decode(self, image):
		#print('**开始解码')
		final_layer_idx = len(self.weight_vars) - 1
		#print('总共%d层卷积'%(final_layer_idx+1))
		out = image
		#print('shape of input:',out.shape)
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			#print('第%d层解码:'%(i+1))
			#print('shape of input:', out.shape)
			if i == 0:
				out = conv2d(out, kernel, bias, dense = False, use_relu = True,
				             Scope = self.scope + '/decoder/b' + str(i), BN = False)
				continue
			if i == final_layer_idx:
				out = conv2d(out, kernel, bias, dense = False, use_relu = False,
				             Scope = self.scope + '/decoder/b' + str(i), BN = False)
				out = tf.nn.tanh(out) / 2 + 0.5
				break
			else:
				out = conv2d(out, kernel, bias, dense = False, use_relu = True, BN = True,
				             Scope = self.scope + '/decoder/b' + str(i))
				continue

			#print('第%d层解码结束:' % (i + 1))
		return out


def conv2d(x, kernel, bias, dense = False, use_relu = True, Scope = None, BN = True):
	# padding image with reflection mode
	#print('  卷积前形状：',x.shape)
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(x_padded, kernel, strides = [1, 1, 1, 1], padding = 'VALID')
	out = tf.nn.bias_add(out, bias)

	if BN:
		with tf.variable_scope(Scope):
			out = tf.layers.batch_normalization(out, training = True)
	if use_relu:
		out = tf.nn.relu(out)
	if dense:
		out = tf.concat([out, x], 3)
	#print('  卷积后的形状：', out.shape)
	#print('卷积结束')
	return out

def up_sample(x, scale_factor = 2):
	_, h, w, _ = x.get_shape().as_list()
	new_size = [h * scale_factor, w * scale_factor]
	return tf.image.resize_nearest_neighbor(x, size = new_size)