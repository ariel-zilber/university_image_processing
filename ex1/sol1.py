import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from skimage.color import rgb2gray

# the type of representations
GRAY_SCALE = 1
RGB = 2

# the matrix to transform from rgb to yiq representation
RGB_TO_YIQ_MATRIX = [
    [0.299, 0.587, 0.114],
    [0.596, -0.275, -0.321],
    [0.212, -0.523, 0.311]
]

# the matrix to transform from yiq to rgb representation
YIQ_TO_RGB_MATRIX = [
    [1, 0.956, 0.621],
    [1, -0.272, -0.647],
    [1, -1.105, 1.702]
]


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


def rgb2yiq(imRGB):
    '''
    Returns a copy of a give RGB image in the YIQ Format
    :param imRGB: a RGB Image which is represents as matix in the format height x width x 3 np.float64.
    The values range  between 0 and 1.
    :return: a copy of a give RGB image in the YIQ Format and in the same dimension
    '''

    # Create a new empty matrix
    image_yiq = np.zeros(imRGB.shape, np.float64)

    # put the Y channel
    image_yiq[:, :, 0] = np.dot(imRGB, RGB_TO_YIQ_MATRIX[0])

    # put the I channel
    image_yiq[:, :, 1] = np.dot(imRGB, RGB_TO_YIQ_MATRIX[1])

    # put the Q channel
    image_yiq[:, :, 2] = np.dot(imRGB, RGB_TO_YIQ_MATRIX[2])

    # makes sure the values are in the appropriate range
    image_yiq[image_yiq > 1] = 1
    image_yiq[image_yiq < 0] = 0

    return image_yiq


def yiq2rgb(imYIQ):
    '''
    Returns a copy of a give YIQ image in the RGB Format
    :param imYIQ: a YIQ Image which is represents as matix in the format height x width x 3 np.float64.
    The Y channel is in the [0,1] range,the I and Q channels are in the [-1, 1] range.
    :return: a copy of a give YIQ image in the RGB Format and in the same dimension
    '''''

    # Create a new empty matrix
    image_rgb = np.zeros(imYIQ.shape, np.float64)

    # put the Y channel
    image_rgb[:, :, 0] = np.dot(imYIQ, RGB_TO_YIQ_MATRIX[0])

    # put the I channel
    image_rgb[:, :, 1] = np.dot(imYIQ, RGB_TO_YIQ_MATRIX[1])

    # put the Q channel
    image_rgb[:, :, 2] = np.dot(imYIQ, RGB_TO_YIQ_MATRIX[2])

    # makes sure the values are in the appropriate range
    image_rgb[image_rgb > 1] = 1
    image_rgb[image_rgb < 0] = 0

    return image_rgb


def get_histogram(img):
    '''
    Returns the histogram using the function np.histogram of numpy
    :param img: the image to create histogram from
    :return: the histogram of the image
    '''
    return np.histogram(img, 256, [0, 1])[0]


def get_normalized_image(image):
    '''
    Normalize a given image
    :param image:
    :return: the image normalized
    '''
    return image.astype(np.float64) / 255


def histogram_equalize(im_orig):
    '''
    performs histogram equalization of a given grayscale or RGB image
    :param im_orig:  the input grayscale or RGB float64 image with values in [0, 1]
    :return:a list [im_eq, hist_orig, hist_eq] where
    im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
    hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
    hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    '''

    def _equalize_histogram(image, hist_orig):
        cumulative_histogram = np.cumsum(hist_orig)
        cumulative_histogram = (cumulative_histogram * 255) / cumulative_histogram[-1]

        image = np.interp(image, np.linspace(0, 1, 256), np.round(cumulative_histogram))
        image_normalized = get_normalized_image(image)

        return image_normalized

    image_dims = im_orig.ndim

    # performs the appropriate action based on the representation(rgb or grayscale)
    if image_dims == 3:
        def _histogram_equalize_rgb(im_orig):
            image_yiq = rgb2yiq(im_orig)
            hist_orig = get_histogram(image_yiq[:, :, 0])

            image_yiq[:, :, 0] = _equalize_histogram(image_yiq[:, :, 0], hist_orig)

            hist_eq = get_histogram(image_yiq[:, :, 0])
            im_eq = yiq2rgb(image_yiq)

            return [im_eq, hist_orig, hist_eq]

        hist_func = _histogram_equalize_rgb
    else:
        def _histogram_equalize_yiq(im_orig):

            image = im_orig.copy()
            hist_orig = get_histogram(image)
            im_eq = _equalize_histogram(image, hist_orig)
            hist_eq = get_histogram(im_eq)

            return [im_eq, hist_orig, hist_eq]

        hist_func = _histogram_equalize_yiq

    equalized_histogram = hist_func(im_orig)

    return equalized_histogram


def quantize(im_orig, n_quant, n_iter):
    '''
     Performs optimal quantization of a given grayscale or RGB image
    :param im_orig: the input grayscale or RGB image to be quantized (float64 image with values in [0, 1])
    :param n_quant: the number of intensities your output im_quant image should have.
    :param n_iter: the maximum number of iterations of the optimization procedure (may converge earlier.)
    :return: a list [im_quant, error] where
    '''


    # helper methods

    def _calculate_z_initial(histogram, intensities_number):

        # create segments to the histogram
        cumulative_histogram = np.cumsum(histogram).astype(np.float64)
        cumulative_histogram = np.rint((cumulative_histogram / np.max(cumulative_histogram)) * (intensities_number - 1))
        segments = np.flatnonzero(np.r_[1, np.diff(cumulative_histogram)[:-1]])

        # returns the combined segments
        return np.append(segments, [255]).astype(np.float64)

    # calculates the z variable using the formula provided in the lectures
    def _calculate_z(q, z):
        for i in range(1, len(q)):
            # using the median value
            z[i] = (q[i] + q[i - 1]) / 2

        return z

    # calculates the q variable using the formula provided in the lectures
    def _calculate_q(z, q, histogram):
        for i in range(len(z) - 1):
            # calculate the range
            segment_start = np.round(z[i]).astype(np.uint8)
            segment_end = np.round(z[i + 1]).astype(np.uint8)

            q[i] = np.sum(histogram[segment_start:segment_end] * np.arange(segment_start, segment_end)) / \
                np.sum(histogram[segment_start:segment_end])

        return q

    def _calculate_error(q, z, histogram):
        total_error = 0

        for i in range(0, len(z) - 1):

            # calculate the range
            segment_start = np.round(z[i]).astype(np.uint8)
            segment_end = np.round(z[i + 1]).astype(np.uint8)

            # calculate the segment error sum
            current_segment_error = np.sum(
                histogram[segment_start:segment_end] * (q[i] - np.arange(segment_start, segment_end)) ** 2)

            # add to the total error
            total_error += current_segment_error

        return total_error


    # create a lookup table for the images
    def _create_lookup_table(z, q):

        # inits an empty lookup table
        lookup_table = np.zeros(256)

        for i in range(len(z) - 1):
            # calculate the boundaries
            begin = int(np.round(z[i]))
            end = int(z[i + 1]) + 1

            lookup_table[begin:end] = q[i]

        return lookup_table

    # check the validity of the input
    if n_quant <= 0 or n_iter <= 0:
        raise ValueError("The given values for n_quant and n_iter must be positive!")

    # checks whenever the image is rgb or grayscale
    if im_orig.ndim == 3:
        image = rgb2yiq(im_orig)
        img_hist = get_histogram(image[:, :, 0])
    else:
        image = im_orig.copy()
        img_hist = get_histogram(image)

    # calculate initial values
    q = np.zeros(n_quant).astype(np.float64)
    z = _calculate_z_initial(img_hist, n_quant)
    error = []

    last_z_iteration = z.copy()

    # calculate the next values for z,q ac
    for i in range(n_iter):
        q = _calculate_q(z, q, img_hist)
        z = _calculate_z(q, z)

        # store the current total SSD error in a list
        error.append(_calculate_error(z, q, img_hist))

        # convergence check
        if np.array_equal(z, last_z_iteration):
            break

        last_z_iteration = z.copy()

    lookup_table = _create_lookup_table(z, q)
    image_dims=im_orig.ndim

    # returns the quantized image

    if image_dims == 3:
        # map key to value using the look up table
        image[:, :, 0] = get_normalized_image(lookup_table[np.rint(image[:, :, 0] * 255).astype(np.uint8)])
        im_quant = yiq2rgb(image)
    else:
        # map key to value using the look up table
        im_quant = get_normalized_image(lookup_table[np.rint(image * 255).astype(np.uint8)])

    return [im_quant, error]



