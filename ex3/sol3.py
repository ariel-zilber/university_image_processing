import os

import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from skimage.color import rgb2gray
from scipy.signal import convolve2d

GRAY_SCALE = 1
RGB = 2

RGB_TO_YIQ_MATRIX = [
    [0.299, 0.587, 0.114],
    [0.596, -0.275, -0.321],
    [0.212, -0.523, 0.311]
]
YIQ_TO_RGB_MATRIX = [
    [1, 0.956, 0.621],
    [1, -0.272, -0.647],
    [1, -1.105, 1.702]
]


def get_normalized_image(image):
    return image.astype(np.float64) / 255


def read_image(filename, representation):
    '''
    Reads an image file and converts it into a given representation.
    :param filename: the filename of an image on disk (could be grayscale or RGB)
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
     image (1) or an RGB image (2). If the input image is grayscale, we will  not  call it with representation = 2
    :return: an  image in the specified  representation
    '''

    image = imread(filename)
    image_normalized = get_normalized_image(image)

    # select a possible representation
    if (representation == GRAY_SCALE):
        return rgb2gray(image_normalized)
    elif (representation == RGB):
        return image_normalized
    else:
        raise ValueError(
            "Invalid representation was give."
            "Possible representations are GRAY_SCALE=1 or RGB=2")


def imdisplay(filename, representation):
    '''
    Displays an image in a given representation
    :param filename: the filename of an image on disk (could be grayscale or RGB)
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    '''

    image = read_image(filename, representation)
    plt.figure()

    # Give the axises titles
    plt.xlabel("image width")
    plt.ylabel("image length")

    # act by the appropriate representation
    if representation == GRAY_SCALE:
        plt.title("Displaying the grayScale image:" + filename)
        plt.imshow(image, cmap='gray')

    else:
        plt.title("Displaying the RGB image:" + filename)
        plt.imshow(image)

    plt.show()

def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)
# 3.1 Gaussian & Laplacian pyramid construction


def binominal_coefficients_vector(n, normalized=False):
    '''

    :param n:
    :return: vector of shape (n,1) with binominal coefficients
    '''

    line = np.array([1, 1])

    i = 1
    while i < n:
        line_t = np.transpose(line)
        line = np.convolve(line, line_t)
        i += 2

    reshaped = line.reshape((1, line.shape[0]))

    if normalized:
        return reshaped / reshaped.sum()
    else:
        return reshaped


def smaller(image):
    new_image = image.copy()
    new_image = new_image[::2, ::2]

    return new_image


def reduce(image, filter_vector):
    out = convolve2d(image, filter_vector, 'same', 'symm')
    out = convolve2d(out, filter_vector.transpose(), 'same', 'symm')
    out = smaller(out)

    return out


def larger(image):
    def insert_zeros(a, N=1):
        # a : Input array
        # N : number of zeros to be inserted between consecutive rows and cols
        out = np.zeros((N + 1) * np.array(a.shape) - N + 1, dtype=a.dtype)
        out[::N + 1, ::N + 1] = a
        return out

    new_image = insert_zeros(image, 1)

    return new_image


def expand(image, filter_vector):
    # 1. zero padding
    out = larger(image)

    # 2. blur
    out = convolve2d(out, filter_vector * 2, 'same', 'wrap')
    out = convolve2d(out, filter_vector.transpose() * 2, 'same', 'wrap')

    return out


def build_gaussian_pyramid(im, max_levels, filter_size):
    '''
    :param im: a grayscale image with double values in [0,1]
    ( the output of ex1's read_image woth representation set to 1)
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the gaussian filter
    :return:
     pyr - the resulting pyramid with maximum length of max_levels
     where each element of the array is a grayscale image
     filter_vec - a row vector of shape (1,filter_size) used for the pyramid construction
    '''

    # find the filter vector
    filter_vec = binominal_coefficients_vector(filter_size, normalized=True)

    # create the pyramid
    layer = im.copy()
    pyr = [layer]

    for i in range(max_levels - 1):
        layer = reduce(layer, filter_vec)
        pyr.append(layer)

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    '''
    :param im:   a grayscale image with double values in [0,1]
    ( the output of ex1's read_image woth representation set to 1)
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the gaussian filter
    :return:
     pyr - the resulting pyramid with maximum length of max_levels
     where each element of the array is a grayscale image
     filter_vec - a row vector of shape (1,filter_size) used for the pyramid construction
    '''

    gaussian_pyramid, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    filter_vec = binominal_coefficients_vector(filter_size, normalized=True)

    layer = gaussian_pyramid[len(gaussian_pyramid) - 1]

    pyr = [layer]

    for i in range(max_levels - 1, 0, -1):
        gaussian_expanded = expand(gaussian_pyramid[i], filter_vec)
        laplacian = np.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
        pyr.insert(0, laplacian)

    return pyr, filter_vec


# 3.2
def laplacian_to_image(lpyr, filter_vec, coeff):
    '''

    :param lpyr: laplacian pyramid generated by the second function in 3.1
    :param filter_vec: laplacian pyramid generated by the second function in 3.1
    :param coeff: a python list.The list length is the same as the number if levels in the pyramid lpyr

    :return: The constructed image
    '''

    max_levels = len(lpyr)
    img = lpyr[max_levels - 1]

    for i in range(max_levels - 2, -1, -1):
        img = lpyr[i] + expand(img, filter_vec) * coeff[i]

    return img


# 3.3


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb

    new_img = np.zeros(shape=(max_height, total_width))
    new_img -= 1

    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb

    return new_img


def render_pyramid(pyr, levels):
    '''

    :param pyr: either a Gaussian or Laplacian pyramid
    :param levels: is the number of levels  to present in the result<=max_levels
    :return res: The image rendered
    '''

    res = pyr[0]

    for i in range(1, levels):
        res = concat_images(res, pyr[i])

    return res


def display_pyramid(pyr, levels):
    rendered_pyramid = render_pyramid(pyr, levels)
    plt.imshow(rendered_pyramid, cmap='gray')
    plt.show()


# 3.4
def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    '''
     :param im1 input grayscale image to be blended
     :param im2 input grayscale image to be blended
     :param mask boolean mask containing True and False representing which parts
     of im1 and im2 should appear in the resulting im_blend
    :param max_levels the max_levels parameter you should use when generating the Gaussian and Laplacian
            pyramids
    :param filter_size_im  the size of the Gaussian filter (an odd scalar that represents a squared filter) which
           defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask
    '''

    im_blend = []

    # laplacian 1
    L1, vec1 = build_laplacian_pyramid(im1, max_levels, filter_size_mask)
    L2, vec2 = build_laplacian_pyramid(im2, max_levels, filter_size_mask)

    # gausian 1
    GM, vectMask = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_im)
    for k in range(0, max_levels):
        im_blend.append(np.multiply(GM[k], L1[k]) + (1 - GM[k]) * L2[k])

    return laplacian_to_image(im_blend, vec1, np.ones(max_levels + 1))


def get_channels(im):
    '''
    Returens each channel form given RGB image
    :param im:
    :return:
    '''
    r = np.zeros(shape=(im.shape[0], im.shape[1]))
    r[:, :] = im[:, :, 0]

    g = np.zeros(shape=(im.shape[0], im.shape[1]))
    g[:, :] = im[:, :, 1]

    b = np.zeros(shape=(im.shape[0], im.shape[1]))
    b[:, :] = im[:, :, 2]

    return r, g, b


def combine_channels(r, g, b, shape):
    '''
    combine data of rgb channel of image to an image
    :param r:
    :param g:
    :param b:
    :param shape:
    :return:
    '''
    im = np.zeros(shape=shape)
    im[:, :, 0] = r[:, :]
    im[:, :, 1] = g[:, :]
    im[:, :, 2] = b[:, :]
    return im


def blend_rgb(img1, img2, mask, max_levels, filter_size_im, filter_size_mask):
    '''
    performs blending on rgb images
    :param img1:
    :param img2:
    :param mask:
    :param max_levels:
    :param filter_size_im:
    :param filter_size_mask:
    :return:
    '''
    r1, g1, b1 = get_channels(img1)
    r2, g2, b2 = get_channels(img2)
    r3, g3, b3 = get_channels(mask)
    blended_r = pyramid_blending(r1, r2, r3, max_levels, filter_size_im, filter_size_mask)
    blended_g = pyramid_blending(g1, g2, g3, max_levels, filter_size_im, filter_size_mask)
    blended_b = pyramid_blending(b1, b2, b3, max_levels, filter_size_im, filter_size_mask)
    b_img = combine_channels(blended_r, blended_g, blended_b, img1.shape)
    b_img = np.clip(b_img, 0, 1)
    return b_img


def plot_blend_result(img1, img2, mask, result):
    # set the axes
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    #  set the title
    ax1.set_title("First Image:")
    ax2.set_title("Second Image:")
    ax3.set_title("Mask Image:")
    ax4.set_title("Blended Image:")

    # set the content
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax3.imshow(mask)
    ax4.imshow(result)

    # display the result
    plt.show()


def blending_example1():
    '''
    An example blending on found images
    :return:im1, im2, mask, im_blend
    '''

    img1 = read_image(relpath("externals/A.jpg"), RGB).astype(np.float64)
    img2 = read_image(relpath("externals/1.jpg"), RGB).astype(np.float64)
    mask = read_image(relpath("externals/mask1.jpg"), RGB).astype(np.bool)

    b_img = blend_rgb(img1, img2, mask, 4, 6, 3)
    b_img=b_img.clip(0,1)

    plot_blend_result(img1, img2, mask.astype(np.float64), b_img)

    return img1,img2,mask,b_img


def blending_example2():
    '''
    An example blending on found images
    :return:im1, im2, mask, im_blend
    '''
    img1 = read_image(relpath("externals/B.jpg"), RGB).astype(np.float64)
    img2 = read_image(relpath("externals/2.jpg"), RGB).astype(np.float64)
    mask = read_image(relpath("externals/mask2.jpg"), RGB).astype(np.bool)
    b_img = blend_rgb(img1, img2, mask, 4, 6, 3)
    b_img=b_img.clip(0,1)
    plot_blend_result(img1, img2, mask.astype(np.float64), b_img)

    return img1,img2,mask,b_img

