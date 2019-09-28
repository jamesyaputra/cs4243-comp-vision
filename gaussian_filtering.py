import numpy as np
from skimage import io
import os.path as osp

def resize(image, new_width, new_height):
    new_image = np.zeros((new_height, new_width, 3), dtype='uint8')
    if len(image.shape)==2:
        new_image = np.zeros((new_height, new_width), dtype='uint8')
    y_scale = new_height / (image.shape[0])
    x_scale = new_width / (image.shape[1])

    for i in range(new_height):
        for j in range(new_width):
            new_image[i][j] = image[int(i/y_scale), int(j/x_scale)]

    return new_image

def rgb2grey(image):
    if len(image.shape) != 3:
        print('Image should have 3 channels')
        return

    weights = [0.299, 0.587, 0.114]
    height, width, depth = image.shape
    greyscale = np.zeros((height, width))
    for channel in range(depth):
        for i in range(height):
            for j in range(width):
                greyscale[i][j] +=  weights[channel] * image[i][j][channel]

    image = greyscale

    return image/255.

def rotate180(kernel):
    """
     5 points
    Rotate the matrix by 180
    :param kernel:
    :return:
    """

    height, width = kernel.shape
    rotated_kernel = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            rotated_kernel[i][j] = kernel[height - i - 1][width - j - 1]
    
    kernel = rotated_kernel

    return kernel

def cs4243_guassian_kernel(ksize, sigma):
    """
     10 points
    Implement the simplified Guassian kernel below:
    k(x,y)=exp((x^2+y^2)/(-2sigma^2))
	Note that Guassian kernel should be central symmentry.
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    """
    kernel = np.zeros((ksize, ksize), dtype=np.float64)

    for x in range(ksize):
        for y in range(ksize):
        	if ksize % 2 == 1:
        		kernel[x][y] = np.exp(((x - ksize//2)**2 + (y - ksize//2)**2) / (-2 * (sigma ** 2)))
        	else:
        		kernel[x][y] = np.exp(((x - ksize//2 + 1)**2 + (y - ksize//2 + 1)**2) / (-2 * (sigma ** 2)))

    return kernel / kernel.sum()

def convolution_naive(image, kernel):
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    h = Hk//2
    w = Wk//2
    
    for x in range(Hi):
        for y in range(Wi):
            for i in range(Hk):
                for j in range(Wk):
                    if (x + h - i) >= 0 and (y + w - j) >= 0 and (x + h - i) < Hi and (y + w - j) < Wi:
                        filtered_image[x][y] += kernel[i][j] * image[x + h - i][y + w - j]

    return filtered_image

def pad_zeros(image, pad_height, pad_width):
    height, width = image.shape
    new_height, new_width = height+pad_height*2, width+pad_width*2
    padded_image = np.zeros((new_height, new_width))
    padded_image[pad_height:new_height-pad_height, pad_width:new_width-pad_width] = image
    return padded_image

def filter_fast(image, kernel):
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    h = Hk//2
    w = Wk//2

    image = pad_zeros(image, h, w)
    kernel = rotate180(kernel)
    for x in range(Hi):
        for y in range(Wi):
            for i in range(Hk):
                for j in range(Wk):
                    filtered_image[x][y] += kernel[i][j] * image[x + i][y + j]

    return filtered_image

def filter_faster(image, kernel):
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    h = Hk//2
    w = Wk//2

    image = pad_zeros(image, h, w)
    kernel = rotate180(kernel)

    matrix = np.zeros((Hi * Wi, Hk * Wk))
    for i in range(Hi * Wi):
    	row = i // Hi
    	col = i % Wi
    	matrix[i, :] = image[row:row + Hk, col:col + Wk].reshape(1, Hk * Wk)

    kernel = kernel.reshape(Hk * Wk, 1)
    filtered_image = np.dot(matrix, kernel).reshape(Hi, Wi)
    return filtered_image

def downsample(image, ratio):
    Hi, Wi = image.shape
    downsample_image = np.zeros((round(Hi / ratio), round(Wi / ratio)))
    downsample_image = image[::ratio, ::ratio]

    return downsample_image