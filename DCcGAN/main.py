from __future__ import print_function
import time
# from utils import list_images
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
from generate import generate
from information_measure import feature_extract
import scipy.ndimage


#训练参数

EPOCHES = 3
BATCH_SIZE = 16
LOGGING = 10
LR = 0.0002  #learning rate
MODEL_SAVE_PATH = '/home/ydx/DDCGAN/model/exp165/'
PATH_LOSS = '/home/ydx/DDCGAN/model/exp165/'
IS_TRAINING = True
#FEATURE_EXTRACT = True
LAYER_INDEX = [0,1,2,3]


no_pics = [9,17,18,22,23,36,37,40,42,44,46]

#f = h5py.File('/home/ydx/datasets/Brain_YUV_16stride.h5', 'r')  #medical
#f = h5py.File('/home/ydx/DDCGAN/Training_Dataset.h5', 'r')   #TNO
#f = h5py.File('/home/ydx/datasets/multisensor_16stride.h5', 'r')   #multisensor
#f = h5py.File('/home/ydx/datasets/ROAD_RGB_16stride.h5', 'r')   #ROAD_RGB
f = h5py.File('/home/ydx/datasets/ROAD_20_64.h5', 'r')   #ROAD  63768
# # for key in f.keys():
# #   print(f[key].name)
sources = f['data'][:]
sources = np.transpose(sources, (0, 3, 2, 1))

def main():
	if IS_TRAINING:
		print(('\nBegin to train the network ...\n'))
		train(sources, MODEL_SAVE_PATH, EPOCHES, BATCH_SIZE,PATH_LOSS,
			  logging_period = LOGGING,LEARNING_RATE=LR,alpha=0.6)

	else :
		print('\nBegin to generate pictures ...\n')
		#path = '/home/ydx/DDCGAN/same-resolution vis-ir image fusion/test_imgs/'
		#savepath = '/home/ydx/DDCGAN/results/exp157/'
		#path = '/home/ydx/datasets/RoadScene_selected/'
		#savepath = '/home/ydx/DDCGAN/results/RoadScene/exp133/'
		#path = '/home/ydx/datasets/CVC14_SELECTED/'
		#savepath = '/home/ydx/DDCGAN/results/CVC14/exp133/'
		path = '/home/ydx/datasets/RoadScene_selected/YCbCr/'
		#savepath = '/home/ydx/DDCGAN/results/RoadScene/YCbCr/exp125/'
		#path = '/home/ydx/datasets/multisensor_selected/'
		savepath = '/home/ydx/DDCGAN/results/exp164/'



		# for root, dirs, files in os.walk(path):
		# 	test_num = len(files)
		indexir = ['a','b','c']
		indexvis = ['a', 'b', 'c']
		Time = []
		for i in range(28):
		#for i in range(127):
			index = i + 1

			#ir_path = path + 'IR' + str(index) +'.bmp'   #20  TNO
			#vis_path = path + 'VIS' + str(index) +'.bmp'
			#ir_path = path + 'IR/' + str(index) + '.jpg'   #28   ROAD
			#vis_path = path + 'VIS_GRAY/' + str(index) +'.jpg'
			#ir_path = path + 'IR/' + str(index) + '.jpg'  #29  CVC
			#vis_path = path + 'VIS/' + str(index) + '.jpg'
			ir_path = path + 'IR/' + str(index) + '.jpg'   #28   ROAD_RGB
			vis_path = path + 'VIS_Y/' + str(index) +'.jpg'
			#ir_path = '/home/ydx/datasets/MEDICAL/PET_Y/' + str(index) + '.jpg'   #127  MEDICAL
			#vis_path = '/home/ydx/datasets/MEDICAL/MRI_GRAY/' + str(index) +'.jpg'
			#ir_path = path + 'IR/' + str(index) + '.jpg'  # 29  multisensor
			#vis_path = path + 'VIS/' + str(index) + '.jpg'


				# if index in no_pics:
				# continue

			model_path = '/home/ydx/DDCGAN/model/exp164/epoch3/' + 'epoch3batch270.ckpt'
			'''			
				for layer in range(4):
					feature_extract(ir_path, vis_path, model_path, index,
						output_path=savepath, layer_index=layer)
					'''
			print("pic_num:%s" % index)
			begin = time.time()
			generate(ir_path, vis_path,model_path, index, output_path=savepath)
			end = time.time()
			Time.append(end - begin)

		print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))


	'''else:
		print('\nBegin to generate pictures ...\n')
		path = '/home/ydx/DDCGAN/same-resolution vis-ir image fusion/test_imgs/'
		savepath = '/home/ydx/DDCGAN/results/exp5/'
		# for root, dirs, files in os.walk(path):
		# 	test_num = len(files)

		Time=[]
		for i in range(20):
			index = i + 1
			ir_path = path + 'IR' + str(index) + '.bmp'
			vis_path = path + 'VIS' + str(index) + '.bmp'
			#if index in no_pics:
				#continue
			begin = time.time()
			model_path = '/home/ydx/DDCGAN/model/exp5/epoch30/' + 'epoch30.ckpt'
			generate(ir_path, vis_path, model_path, index, output_path = savepath)
			end = time.time()
			Time.append(end - begin)
			print("pic_num:%s" % index)
		print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))'''


if __name__ == '__main__':
	main()
