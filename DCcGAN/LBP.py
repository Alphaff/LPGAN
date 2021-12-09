import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import tensorflow as tf
import os




sess = tf.Session()


def getHopTimes(data):
    '''
    计算跳变次数
    '''
    count = 0
    binaryCode = "{0:0>8b}".format(data)

    for i in range(1, len(binaryCode)):
        if binaryCode[i] != binaryCode[(i - 1)]:
            count += 1
    return count


def uniform_pattern_LBP(img, radius=3, neighbors=8):
    h, w = img.shape
    dst = tf.Variable(tf.zeros([h - 2 * radius, w - 2 * radius]))
    # LBP特征值对应图像灰度编码表，直接默认采样点为8位
    temp = 1
    table = tf.Variable(tf.zeros([256]))
    for i in range(256):
        if getHopTimes(i) < 3:
            table[i].assign(temp)
            temp += 1
    # 是否进行UniformPattern编码的标志
    flag = False
    # 计算LBP特征图  相当于事先构建了一个map
    for k in range(neighbors):
        if k == neighbors - 1:
            flag = True

        # 计算采样点对于中心点坐标的偏移量rx，ry
        rx = radius * np.cos(2.0 * np.pi * k / neighbors)
        ry = -(radius * np.sin(2.0 * np.pi * k / neighbors))
        # 为双线性插值做准备
        # 对采样点偏移量分别进行上下取整
        x1 = int(np.floor(rx))
        x2 = int(np.ceil(rx))
        y1 = int(np.floor(ry))
        y2 = int(np.ceil(ry))
        # 将坐标偏移量映射到0-1之间
        tx = rx - x1
        ty = ry - y1
        # 根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty
        # 循环处理每个像素
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                # 获得中心像素点的灰度值
                center = img[i, j]
                # 根据双线性插值公式计算第k个采样点的灰度值
                neighbor = img[i + y1, j + x1] * w1 + img[i + y2, j + x1] * w2 + \
                           img[i + y1, j + x2] * w3 + img[i + y2, j + x2] * w4
                # LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                dst[i - radius, j - radius] |= (np.uint8)(neighbor > center) << (np.uint8)(neighbors - k - 1)
                # 进行LBP特征的UniformPattern编码
                if flag:
                    dst[i - radius, j - radius] = table[dst[i - radius, j - radius]]
    return dst


def ori_lbp(img):
	kernel1 = tf.constant([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
	kernel1 = tf.expand_dims(kernel1, axis=-1)
	kernel1 = tf.expand_dims(kernel1, axis=-1)
	kernel2 = tf.constant([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
	kernel2 = tf.expand_dims(kernel2, axis=-1)
	kernel2 = tf.expand_dims(kernel2, axis=-1)
	kernel3 = tf.constant([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
	kernel3 = tf.expand_dims(kernel3, axis=-1)
	kernel3 = tf.expand_dims(kernel3, axis=-1)
	kernel4 = tf.constant([[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
	kernel4 = tf.expand_dims(kernel4, axis=-1)
	kernel4 = tf.expand_dims(kernel4, axis=-1)
	kernel5 = tf.constant([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
	kernel5 = tf.expand_dims(kernel5, axis=-1)
	kernel5 = tf.expand_dims(kernel5, axis=-1)
	kernel6 = tf.constant([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
	kernel6 = tf.expand_dims(kernel6, axis=-1)
	kernel6 = tf.expand_dims(kernel6, axis=-1)
	kernel7 = tf.constant([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
	kernel7 = tf.expand_dims(kernel7, axis=-1)
	kernel7 = tf.expand_dims(kernel7, axis=-1)
	kernel8 = tf.constant([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
	kernel8 = tf.expand_dims(kernel8, axis=-1)
	kernel8 = tf.expand_dims(kernel8, axis=-1)
	kernels = [kernel1,kernel2,kernel3,kernel4,kernel5,kernel6,kernel7,kernel8]
	out = tf.zeros_like(img)
	zeros = tf.zeros_like(img)
	ones = tf.ones_like(img)
	for i in range(8):
		out_filter = tf.nn.conv2d(img,kernels[i],strides=[1,1,1,1],padding='SAME',name='ori_lbp')
		cond = out_filter < 0
		out_filter = tf.where(cond, zeros, ones)
		out += out_filter * 2**(i)

	#out = tf.floor(tf.math.log(out) / tf.math.log(2.))
	#cond = out < 0
	#out = tf.where(cond, zeros, out)
	#L = [0.]*256
	L = []
	for i in range(256):
		num = tf.ones_like(out) * i
		#cond = out == i
		#l = tf.where(out == tf.constant([i]), tf.add(l, one), l)
		l = tf.equal(out,num)
		#print('shape of l:',l.shape)
		#l = tf.where(l, tf.constant([1.]), tf.constant([0.]))
		l = tf.cast(l, tf.float32)
		#print('shape of CAST l:', l.shape)
		l = tf.reduce_sum(l, axis=(1,2))
		L.append(l)

	distribution = L[0]
	for i in range(1, 256):
		distribution = tf.concat([distribution,L[i]], axis=1)

	#distribution = distribution / tf.tile(tf.reshape(tf.reduce_sum(distribution, axis=1), [3, 1]), [1,256])
	return distribution #3,256


def lbp(img, cell_size=21):
	#计算整图的LBP
	kernel1 = tf.constant([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
	kernel1 = tf.expand_dims(kernel1, axis=-1)
	kernel1 = tf.expand_dims(kernel1, axis=-1)
	kernel2 = tf.constant([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
	kernel2 = tf.expand_dims(kernel2, axis=-1)
	kernel2 = tf.expand_dims(kernel2, axis=-1)
	kernel3 = tf.constant([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
	kernel3 = tf.expand_dims(kernel3, axis=-1)
	kernel3 = tf.expand_dims(kernel3, axis=-1)
	kernel4 = tf.constant([[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
	kernel4 = tf.expand_dims(kernel4, axis=-1)
	kernel4 = tf.expand_dims(kernel4, axis=-1)
	kernel5 = tf.constant([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
	kernel5 = tf.expand_dims(kernel5, axis=-1)
	kernel5 = tf.expand_dims(kernel5, axis=-1)
	kernel6 = tf.constant([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
	kernel6 = tf.expand_dims(kernel6, axis=-1)
	kernel6 = tf.expand_dims(kernel6, axis=-1)
	kernel7 = tf.constant([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
	kernel7 = tf.expand_dims(kernel7, axis=-1)
	kernel7 = tf.expand_dims(kernel7, axis=-1)
	kernel8 = tf.constant([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
	kernel8 = tf.expand_dims(kernel8, axis=-1)
	kernel8 = tf.expand_dims(kernel8, axis=-1)
	kernels = [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8]
	out = tf.zeros_like(img)
	zeros = tf.zeros_like(img)
	ones = tf.ones_like(img)
	for i in range(8):
		out_filter = tf.nn.conv2d(img, kernels[i], strides=[1, 1, 1, 1], padding='SAME', name='ori_lbp')
		cond = out_filter < 0
		out_filter = tf.where(cond, zeros, ones)
		out += out_filter * 2 ** (i)

	#计算cell的数量和分布
	num_x = img.shape[1] // cell_size
	num_y = img.shape[2] // cell_size

	#计算各个cell的LBP分布
	L = []
	for j in range(num_y):
		for i in range(num_x):
			lbp_cell = out[:,i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size,:]
			for n in range(256):
				num = tf.ones_like(lbp_cell) * n
				# cond = out == i
				# l = tf.where(out == tf.constant([i]), tf.add(l, one), l)
				l = tf.equal(lbp_cell, num)
				#print('shape of l:', l.shape)
				# l = tf.where(l, tf.constant([1.]), tf.constant([0.]))
				l = tf.cast(l, tf.float32)
				#print('shape of CAST l:', l.shape)
				l = tf.reduce_sum(l, axis=(1, 2))
				L.append(l)

	#拼接
	distribution = L[0]
	for i in range(1, len(L)):
		distribution = tf.concat([distribution, L[i]], axis=1)

	return distribution




'''img = tf.constant([[[[10.0],[2.0],[30.0]],[[2.0],[3.0],[4.0]],[[40.0],[5.0],[6.0]]],[[[5.0],[20.0],[3.0]],[[2.0],[3.0],[4.0]],[[4.0],[5.0],[6.0]]],[[[5.0],[2.0],[30.0]],[[2.0],[3.0],[4.0]],[[4.0],[50.0],[6.0]]]])
out, L= ori_lbp(img)
print(sess.run(out))
print('*********************************')

print(sess.run(L[0]))'''
#print(sess.run(out_))

'''img = tf.random_normal([256,256],mean=150,stddev=50)
up_img = uniform_pattern_LBP(img)
print(sess.run(up_img))'''
