import numpy as np
cimport numpy as np

# Terminology:
#   The 'kernal' is a two dimensional matrix of weights, without any bias weight. The bias can and should be added after the fact.
#   The 'img' (image) is the input, a two dimensional matrix which typically represents the pixles of an image.
#	The 'error' is a two dimensional matrix of errors.

# Gets the derivative of the kernal (the weight matrix)
def getDerivative(np.ndarray[np.float64_t, ndim=2] img, np.ndarray[np.float64_t, ndim=2] error, kernal_shape):
    cdef unsigned int krows = kernal_shape[0]
    cdef unsigned int kcols = kernal_shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] dk = np.zeros((krows, kcols))
    cdef unsigned int rows = error.shape[0]
    cdef unsigned int cols = error.shape[1]
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int a
    cdef unsigned int b

    for i in range(rows):
        for j in range(cols):
            for a in range(krows):
                for b in range(kcols):
                    dk[a, b] += img[i+a, j+b] * error[i, j]
    return dk

# Performs the forward pass of the input signal
def conv2d(np.ndarray[np.float64_t, ndim=2] img, np.ndarray[np.float64_t, ndim=2] kernal):
    cdef np.ndarray[np.float64_t, ndim=2] y = np.zeros((img.shape[0] - kernal.shape[0] + 1, img.shape[1] - kernal.shape[1] + 1))
    cdef unsigned int krows = kernal.shape[0]
    cdef unsigned int kcols = kernal.shape[1]
    cdef unsigned int rows = y.shape[0]
    cdef unsigned int cols = y.shape[1]
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int a
    cdef unsigned int b

    for i in range(rows):
        for j in range(cols):
            for a in range(krows):
                for b in range(kcols):
                    y[i, j] += kernal[a, b] * img[i+a, j+b]
    return y
    
# Performs the backward pass of the error signal
def backconv2d(np.ndarray[np.float64_t, ndim=2] e, np.ndarray[np.float64_t, ndim=2] kernal):
    cdef np.ndarray[np.float64_t, ndim=2] y = np.zeros((e.shape[0]+kernal.shape[0]-1, e.shape[1]+kernal.shape[1]-1))
    cdef unsigned int krows = kernal.shape[0]
    cdef unsigned int kcols = kernal.shape[1]
    cdef unsigned int rows = e.shape[0]
    cdef unsigned int cols = e.shape[1]
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int a
    cdef unsigned int b

    for i in range(rows):
        for j in range(cols):
            for a in range(krows):
                for b in range(kcols):
                    y[i+a, j+b] += kernal[a, b] * e[i, j]
    return y