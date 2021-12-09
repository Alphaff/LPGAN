# Train the DenseFuse Net

from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from scipy.misc import imsave
import scipy.ndimage

from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
from LOSS import  L1_LOSS, Fro_LOSS, _tf_fspecial_gauss,L2_LOSS, LOSS_SSIM
from generate import generate
from Generator import Decoder
import math
import os
import glob
from skimage.feature import local_binary_pattern
from LBP import ori_lbp, lbp



patch_size = 64
# TRAINING_IMAGE_SHAPE = (patch_size, patch_size, 2)  # (height, width, color_channels)
EPSILON = 1e-5  #没用
DECAY_RATE = 0.75
eps = 1e-8


#alpha为G_LOSS的两部分的调节权重
def train(source_imgs, save_path, EPOCHES_set, BATCH_SIZE,PATH_LOSS=None,
		  logging_period = 1,LEARNING_RATE = 0.0001,alpha=0.6):
	#from datetime import datetime
	start_time = time.time()
	EPOCHS = EPOCHES_set
	print('Epoches: %d, Batch_size: %d' % (EPOCHS, BATCH_SIZE))

	num_imgs = source_imgs.shape[0]
	print('shape of source imgs:',source_imgs.shape)
	mod = num_imgs % BATCH_SIZE
	n_batches = int(num_imgs // BATCH_SIZE)
	#n_batches = 2
	print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

	if mod > 0:
		print('Train set has been trimmed %d samples...\n' % mod)
		source_imgs = source_imgs[:-mod]

	# create the graph
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)  # 占满资源
	config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True,
							log_device_placement=False)
	#config.gpu_options.allow_growth = True

	with tf.Graph().as_default(), tf.Session(config=config) as sess:
		SOURCE_VIS = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'SOURCE_VIS')
		SOURCE_IR = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'SOURCE_IR')
		print('source_vis shape:', SOURCE_VIS.shape)
		print('source image number:',SOURCE_VIS[0,0,0,:])
		print('source_ir shape:', SOURCE_IR.shape)

		G = Generator('Generator')
		D = Decoder('Decoder')
		#generated_img = G.transform(vis = SOURCE_VIS, ir = SOURCE_IR)
		generated_img1 = G.transform1(vis=SOURCE_VIS, ir=SOURCE_IR,ir_trans=(1-SOURCE_IR),
									  vis_trans=(1-SOURCE_VIS))
		#print('shape of generated_img1:',generated_img1.shape)
		generated_img2 = G.transform2(vis=SOURCE_VIS, ir=SOURCE_IR,ir_trans=(1-SOURCE_IR),
									  vis_trans=(1-SOURCE_VIS))
		#print('shape of generated_img2:', generated_img2.shape)

		#generated_img3 = generated_img1 + generated_img2
		generated_img3 = tf.concat([generated_img1, generated_img2], -1)
		#print('shape of generated_img3:', generated_img3.shape)

		generated_img = D.decode(generated_img3)
		#print('shape of generated_img:', generated_img.shape)

		'''lbp_img = generated_img[0,:,:,0].eval()
		lbp_img = local_binary_pattern(lbp_img, 8, 2, method="nri-uniform")
		lbp_img = tf.constant(lbp_img)
		lbp_img = tf.expand_dims(lbp_img, axis=0)
		lbp_img = tf.expand_dims(lbp_img, axis=-1)
		lbp_vis = SOURCE_VIS[0,:,:,0].eval()
		lbp_vis = local_binary_pattern(lbp_vis, 8, 2, method="nri-uniform")
		lbp_vis = tf.constant(lbp_vis)
		lbp_vis = tf.expand_dims(lbp_vis, axis=0)
		lbp_vis = tf.expand_dims(lbp_vis, axis=-1)'''
		lbp_img = lbp(generated_img)
		lbp_vis = lbp(SOURCE_VIS)

		D1 = Discriminator1('Discriminator1')
		D1_real = D1.discrim(SOURCE_VIS, reuse = False)
		D1_fake = D1.discrim(generated_img, reuse = True)

		D2 = Discriminator2('Discriminator2')
		D2_real = D2.discrim(SOURCE_IR, reuse = False)
		D2_fake = D2.discrim(generated_img, reuse = True)

		#######  LOSS FUNCTION
		# Loss for Generator
		# 求模 角度  VIS
		grad_of_vis = gradmod(SOURCE_VIS)
		grad_of_ir = gradmod(SOURCE_IR)
		angle_vis = grad_direction(SOURCE_VIS)
		angle_ir = grad_direction(SOURCE_IR)

		# grad_angle_vis = grad_direction(SOURCE_VIS)

		# 求模 角度  generated img
		grad_of_img = gradmod(generated_img)
		angle_img = grad_direction(generated_img)

		G_loss_GAN_D1 = -tf.reduce_mean(tf.log(D1_fake + eps))
		G_loss_GAN_D2 = -tf.reduce_mean(tf.log(D2_fake + eps))
		G_loss_GAN = G_loss_GAN_D1 + G_loss_GAN_D2


		LOSS_LBP = tf.reduce_sum(tf.abs(lbp_img - lbp_vis), axis=1)
		LOSS_LBP = tf.reduce_mean(LOSS_LBP)
		#LOSS_LBP = tf.constant([0.0])
		LOSS_IR =  Fro_LOSS(generated_img - SOURCE_IR) #+ 0.2 * Fro_LOSS(generated_img - SOURCE_VIS)
		#LOSS_IR = Fro_LOSS(generated_img - SOURCE_IR)
		LOSS_VIS =  L1_LOSS(grad_of_img - grad_of_vis) #+ 0.2 * L1_LOSS(grad_of_img - grad_of_ir)
		#LOSS_VIS = L1_LOSS(grad_of_img - grad_of_vis)
		LOSS_ANGLE = Fro_LOSS(angle_img - angle_vis)
		#LOSS_ANGLE = tf.constant([0.0])
		LOSS_ssim = LOSS_SSIM(SOURCE_IR,SOURCE_VIS,generated_img)
		#G_loss_norm = 0.6 * (LOSS_IR / 16 + 0.45 * LOSS_VIS + 0.1 * LOSS_ANGLE) + 100 * LOSS_ssim
		#G_loss_norm = LOSS_IR / 16 + 1.2 * LOSS_VIS
		G_loss_norm = 500 * LOSS_ssim + 0.03 * LOSS_IR + 0.2 * LOSS_VIS + 0.5 * LOSS_LBP
		G_loss = G_loss_GAN + alpha * G_loss_norm

		# Loss for Discriminator1
		D1_loss_real = -tf.reduce_mean(tf.log(D1_real + eps))
		D1_loss_fake = -tf.reduce_mean(tf.log(1. - D1_fake + eps))
		D1_loss = D1_loss_fake + D1_loss_real

		# Loss for Discriminator2
		D2_loss_real = -tf.reduce_mean(tf.log(D2_real + eps))
		D2_loss_fake = -tf.reduce_mean(tf.log(1. - D2_fake + eps))
		D2_loss = D2_loss_fake + D2_loss_real

		current_iter = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
		                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
		                                           staircase = False)

		# theta_de = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'deconv_ir')
		theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
		theta_D1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator1')
		theta_D2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator2')

		G_GAN_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss_GAN, global_step = current_iter,
		                                                                 var_list = theta_G)
		G_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss, global_step = current_iter,
		                                                             var_list = theta_G)
		# G_GAN_solver = tf.train.AdamOptimizer(learning_rate).minimize(G_loss_GAN, global_step = current_iter,
		#                                                                  var_list = theta_G)
		D1_solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(D1_loss, global_step = current_iter,
		                                                                      var_list = theta_D1)
		D2_solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(D2_loss, global_step = current_iter,
		                                                                      var_list = theta_D2)

		clip_G = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_G]
		clip_D1 = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D1]
		clip_D2 = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D2]

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep = None)

		tf.summary.scalar('G_Loss_D1', G_loss_GAN_D1)
		tf.summary.scalar('G_Loss_D2', G_loss_GAN_D2)
		tf.summary.scalar('D1_real', tf.reduce_mean(D1_real))
		tf.summary.scalar('D1_fake', tf.reduce_mean(D1_fake))
		tf.summary.scalar('D2_real', tf.reduce_mean(D2_real))
		tf.summary.scalar('D2_fake', tf.reduce_mean(D2_fake))
		tf.summary.image('vis', SOURCE_VIS, max_outputs = 3)
		tf.summary.image('ir', SOURCE_IR, max_outputs = 3)
		tf.summary.image('fused_img', generated_img, max_outputs = 3)

		tf.summary.scalar('Learning rate', learning_rate)
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logs/", sess.graph)

		# ** Start Training **
		step = 0
		count_loss = 0
		num_imgs = source_imgs.shape[0]

		min_loss = 5000
		min_epoch = 0
		min_batch = 0
		for epoch in range(EPOCHS):
			np.random.shuffle(source_imgs)
			for batch in range(n_batches):
				step += 1
				current_iter = step
				VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
				IR_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
				VIS_batch = np.expand_dims(VIS_batch, -1)
				IR_batch = np.expand_dims(IR_batch, -1)
				FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch}

				it_g = 0
				it_d1 = 0
				it_d2 = 0
				# run the training step
				if batch % 2==0:
					sess.run([D1_solver, clip_D1], feed_dict = FEED_DICT)
					it_d1 += 1
					sess.run([D2_solver, clip_D2], feed_dict = FEED_DICT)
					it_d2 += 1
				else:
					sess.run([G_solver, clip_G], feed_dict = FEED_DICT)
					it_g += 1
				g_loss,g_loss_gan, d1_loss, d2_loss, loss_ssim, loss_vis, loss_ir, loss_angle, loss_lbp= \
					sess.run([G_loss,G_loss_GAN, D1_loss, D2_loss, LOSS_ssim, LOSS_VIS, LOSS_IR, LOSS_ANGLE, LOSS_LBP], feed_dict = FEED_DICT)

				if batch%2==0:
					while d1_loss > 1.7 and it_d1 < 20:
						sess.run([D1_solver, clip_D1], feed_dict = FEED_DICT)
						d1_loss = sess.run(D1_loss, feed_dict = FEED_DICT)
						it_d1 += 1
					while d2_loss > 1.7 and it_d2 < 20:
						sess.run([D2_solver, clip_D2], feed_dict = FEED_DICT)
						d2_loss = sess.run(D2_loss, feed_dict = FEED_DICT)
						it_d2 += 1
						d1_loss = sess.run(D1_loss, feed_dict = FEED_DICT)
				else:
					while (d1_loss < 1.4 or d2_loss < 1.4) and it_g < 20:
						sess.run([G_GAN_solver, clip_G], feed_dict = FEED_DICT)
						g_loss,g_loss_gan, d1_loss, d2_loss = sess.run([G_loss,G_loss_GAN, D1_loss, D2_loss], feed_dict = FEED_DICT)
						it_g += 1
					while (g_loss > 800) and it_g < 20:
						sess.run([G_solver, clip_G], feed_dict = FEED_DICT)
						g_loss,g_loss_gan = sess.run([G_loss,G_loss_GAN], feed_dict = FEED_DICT)
						it_g += 1
				#with open(PATH_LOSS+'loss'+'.txt','a') as f:
				print("epoch: %d/%d, batch: %d" % (epoch + 1, EPOCHS, batch+1))


				if (batch+1) % logging_period == 0:
					elapsed_time = time.time() - start_time
					lr = sess.run(learning_rate)
					saver.save(sess, save_path + 'epoch' + str((epoch + 1)) + '/' + 'epoch' +
							   str((epoch + 1)) + 'batch' + str((batch + 1)) + '.ckpt')
					print('epoch: %d/%d, batch: %d,G_loss: %s,G_loss_GAN:%s, D1_loss: %s, D2_loss: %s,LOSS_SSIM: %s,'
						  'LOSS_VIS: %s,LOSS_IR: %s,LOSS_ANGLE: %s,LOSS_LBP: %s' % (
							  epoch + 1, EPOCHS, batch + 1, g_loss,g_loss_gan, d1_loss, d2_loss, loss_ssim, loss_vis, loss_ir,loss_angle,loss_lbp))
					print("lr: %s, elapsed_time: %s\n" % (lr, elapsed_time))

					if g_loss < min_loss:
						min_epoch = str((epoch + 1))
						min_batch = str((batch + 1))
						min_loss = g_loss

					if g_loss < 5000:
						with open(PATH_LOSS +'loss'+ '.txt', 'a') as f:
							print('epoch: %d/%d, batch: %d,G_loss: %s, D1_loss: %s, D2_loss: %s,LOSS_SSIM: %s,'
								  'LOSS_VIS: %s,LOSS_IR: %s,LOSS_ANGLE: %s,LOSS_LBP: %s' % (
								epoch + 1, EPOCHS, batch+1,g_loss, d1_loss, d2_loss,loss_ssim,loss_vis,loss_ir,loss_angle,loss_lbp), file=f)
						with open(PATH_LOSS +'loss'+ '.txt', 'a') as f:
							print("lr: %s, elapsed_time: %s\n" % (lr, elapsed_time), file=f)

				result = sess.run(merged, feed_dict=FEED_DICT)
				writer.add_summary(result, step)
				'''if (step+1) % logging_period == 0:
					saver.save(sess, save_path + 'epoch' + str((epoch + 1)) + '/' + 'epoch' +
							   str((epoch + 1)) + 'batch' + str((batch + 1)) + '.ckpt')
					print('epoch: %d/%d, batch: %d,G_loss: %s, D1_loss: %s, D2_loss: %s,LOSS_SSIM: %s,'
						  'LOSS_VIS: %s,LOSS_IR: %s' % (
						epoch + 1, EPOCHS, batch + 1, g_loss, d1_loss, d2_loss,loss_ssim,loss_vis,loss_ir))
					print("lr: %s, elapsed_time: %s\n" % (lr, elapsed_time))
					if g_loss < 150:
						with open(PATH_LOSS +'loss'+ '.txt', 'a') as f:
							print('epoch: %d/%d, batch: %d,G_loss: %s, D1_loss: %s, D2_loss: %s' % (
								epoch + 1, EPOCHS, batch+1,g_loss, d1_loss, d2_loss), file=f)
						with open(PATH_LOSS +'loss'+ '.txt', 'a') as f:
							print("lr: %s, elapsed_time: %s\n" % (lr, elapsed_time), file=f)'''

				is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)
				if is_last_step or step % logging_period == 0:
					elapsed_time = time.time() - start_time
					lr = sess.run(learning_rate)
					#with open(PATH_LOSS + 'loss'+'.txt', 'a') as f:
					print('epoch:%d/%d, step:%d, lr:%s, elapsed_time:%s' % (
						epoch + 1, EPOCHS, step, lr, elapsed_time))
			writer.close()
			saver.save(sess, save_path + 'epoch'+str((epoch+1)) + '/' + 'epoch'+
				   	str((epoch+1)) + '.ckpt')
		with open(PATH_LOSS + 'loss' + '.txt', 'a') as f:
			print("BEST:epoch:%s, batch:%s\n" % (min_epoch, min_batch), file=f)


def grad(img):
	'''
	求梯度
	'''
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g

def gradxy(img):
	kernelx = tf.constant([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
	kernelx = tf.expand_dims(kernelx, axis=-1)
	kernelx = tf.expand_dims(kernelx, axis=-1)
	gx = tf.nn.conv2d(img, kernelx, strides=[1, 1, 1, 1], padding='SAME')
	kernely = tf.constant([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
	kernely = tf.expand_dims(kernely, axis=-1)
	kernely = tf.expand_dims(kernely, axis=-1)
	gy = tf.nn.conv2d(img, kernely, strides=[1, 1, 1, 1], padding='SAME')
	#g = tf.reduce_sum(tf.sqrt(tf.square(gx, gy)), axis=[1, 2])

	return gx, gy

def gradmod(img):
	gx, gy = gradxy(img)
	g = tf.sqrt(tf.square(gx)+tf.square(gy) + 1e-10)
	#g = tf.abs(gx) + tf.abs(gy)
	return g


def grad_direction(img):
	gx, gy = gradxy(img)
	angle = tf.atan2(gy, gx + 1e-10)
	cond = angle < 0
	pis = tf.ones_like(angle) * math.pi
	return tf.where(cond, angle + pis, angle)


