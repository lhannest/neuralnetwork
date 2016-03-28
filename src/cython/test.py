import pyximport; pyximport.install()
import convolution
import numpy as np
from scipy.signal import convolve2d
import time
import pudb

def _conv2d(img, kernal):
	y = np.zeros((img.shape[0] - kernal.shape[0] + 1, img.shape[1] - kernal.shape[1] + 1))
	for i in range(y.shape[0]):
		for j in range(y.shape[1]):
			for a in range(kernal.shape[0]):
				for b in range(kernal.shape[1]):
					y[i, j] += kernal[a,b] * img[i+a, j+b]
	return y

img = np.array([[1,2,3],[4,5,6],[7,8,9]]) * 1.0
k = np.array([[1,2],[2,3]]) * 1.0

# img = np.random.randn(150, 150)
# k = np.random.randn(9, 7)	

t = time.time()
_conv2d(img, k)
print 'python:', time.time() - t

t = time.time()
convolution.conv2d(img, k)
print 'cython:', time.time() - t

t = time.time()
convolution.deconv2d(img, k)
print 'deconv:', time.time() - t

t = time.time()
convolve2d(img, np.rot90(k, 2), mode='valid')
print 'scipy:', time.time() - t

c = convolution.conv2d(img, k)
print c
print convolve2d(img, np.rot90(k, 2), mode='valid')
print convolution.deconv2d(c, k)
print k