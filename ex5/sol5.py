import random

import numpy as np
import sol5_utils

from tensorflow.keras.layers import Input, Conv2D, Activation, Add
from tensorflow.keras.models import Model
from scipy.ndimage.filters import convolve
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from imageio import imread

# General parameters
SUB_VALUE = 0.5
CONV_KERNEL_SIZE = 3
SPLIT_80_20 = 0.8
BETA_2_PARAM = 0.9
MIN_RAND_ANGLE = 0
MAX_RAND_ANGLE = np.pi
LOSS_FUNCTION = 'mean_squared_error'
BATCH_NUM = 100
STEPS_PER_EPOCH = 100
VALIDATION_SET = 1000

# Denoise related parameters
DENOISE_MIN_SIGMA = 0
DENOISE_MAX_SIGMA = 0.2
DENOISE_PATCH_SIZE = 24
DENOISE_CHANNELS_NUM = 48
DENOISE_TOTAL_EPOCH = 5

# Quick mode related parameters
QUICK_BATCH_NUM = 10
QUICK_STEPS_PER_EPOCH = 3
QUICK_TOTAL_EPOCH = 2
QUICK_VALIDATION_SET = 30

# Deblur related parameters
DEBLUR_KERNEL_SIZE = 7
DEBLUR_PATCH_SIZE = 16
DEBLUR_CHANNELS = 32
DEBLUR_EPOCHS = 10

from skimage.color import rgb2gray

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


def load_dataset(filenames, batch_size, corruption_func,
                 crop_size):
    im_dict = {}

    while True:

        height, width = crop_size
        batch_shape = (batch_size, crop_size[0], crop_size[1], 1)

        target_batch = np.ndarray(batch_shape)
        source_batch = np.ndarray(batch_shape)

        for i in range(batch_size):

            # choose a random image
            random_id = np.random.randint(0, len(filenames) - 1)
            if filenames[random_id] not in im_dict:
                im_dict[filenames[random_id]] = read_image(filenames[random_id], 1)
                im = im_dict[filenames[random_id]]
            else:
                im = im_dict[filenames[random_id]]

            #  Choose size of patch
            h, w = crop_size
            image_height, image_width = im.shape

            if h * 3 > image_height or w * 3 > image_width:
                x_size_large, y_size_large = im.shape
            else:
                x_size_large, y_size_large = height * 3, width * 3

            # Get the patch
            x_im, y_im = im.shape
            random_x = np.random.randint(0, x_im - x_size_large)
            random_y = np.random.randint(0, y_im - y_size_large)
            clear_patch = im[random_x:random_x + x_size_large, random_y:random_y + y_size_large]

            corrupted_patch = corruption_func(clear_patch)
            h_large, w_large = clear_patch.shape
            final_x = np.random.randint(0, h_large - height)
            final_y = np.random.randint(0, w_large - width)

            # Get the the final patch
            final_patch = clear_patch[final_x:final_x + height, final_y: final_y + width]
            final_patch = final_patch[:, :, np.newaxis]

            # Get the corrupted patch
            final_corrupted_patch = corrupted_patch[final_x:final_x + height, final_y:final_y + width]
            final_corrupted_patch = final_corrupted_patch[:, :, np.newaxis]

            # Fill the source batch and the target batch
            source_batch[i] = final_corrupted_patch - SUB_VALUE
            target_batch[i] = final_patch - SUB_VALUE

        yield source_batch, target_batch


def resblock(input_tensor, num_channels):
    '''
    Takes as input a symbolic input tensor and the number of channels for each of its convolutional layers,
     and returns the symbolic output tensor of the layer configuration
    :param input_tensor: symbolic tensor
    :param num_channels: number of channels for the convolution layers
    :return:symbolic output tensor of the layer
    '''
    a = Conv2D(num_channels, (CONV_KERNEL_SIZE, CONV_KERNEL_SIZE), padding='same')(input_tensor)
    b = Activation("relu")(a)
    c = Conv2D(num_channels, (CONV_KERNEL_SIZE, CONV_KERNEL_SIZE), padding='same')(b)
    output = Add()([input_tensor, c])
    output = Activation('relu')(output)
    return output


def build_nn_model(height, width, num_channels, num_res_blocks):
    '''
        Returns an untrained Keras model of a ResNet network.
    :param height: The height of the input image
    :param width:The width of the input image
    :param num_channels: The number of channels used for the convolution layer
    :param num_res_blocks: The number of residual blocks
    :return: Untrained keras model
    '''

    a = Input(shape=(height, width, 1))
    b = Conv2D(num_channels, (CONV_KERNEL_SIZE, CONV_KERNEL_SIZE), padding='same')(a)
    c = Activation("relu")(b)
    d = c

    for i in range(num_res_blocks):
        d = resblock(d, num_channels)

    output = Add()([d, c])
    output = Conv2D(1, (CONV_KERNEL_SIZE, CONV_KERNEL_SIZE), padding='same')(output)

    return Model(inputs=a, outputs=output)


def train_model(model, images, corruption_func,
                batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    '''
        Trains an input neural network
    :param model: The keras model of the network
    :param images: Path lists to use as dataset of images
    :param corruption_func: The function used to corrupt the dataset
    :param batch_size: The size of the given batch
    :param steps_per_epoch: The number of samples used in each epoch
    :param num_epochs: The number of epochs that are going to be used by the optimization
    :param num_valid_samples: The number of validation steps after every epoch
    :return: A trained model
    '''

    crop_size = model.input_shape[1: 3]

    training_set_len = int(len(images) * SPLIT_80_20)
    training_images = images[: training_set_len]
    validation_images = images[training_set_len:]

    training_set_data = load_dataset(training_images, batch_size, corruption_func, crop_size)
    validation_set_data = load_dataset(validation_images, batch_size, corruption_func, crop_size)

    adam = Adam(beta_2=BETA_2_PARAM)
    model.compile(loss=LOSS_FUNCTION, optimizer=adam)

    model.fit_generator(training_set_data, steps_per_epoch=steps_per_epoch,
                        epochs=num_epochs, validation_data=validation_set_data,
                        validation_steps=num_valid_samples)


def restore_image(corrupted_image, base_model):
    '''
    Restores a corrupted image
    :param corrupted_image: A corrupted image of type float64 in range [0,1]
    :param base_model: A trained keras model
    :return: The restored image
    '''

    # The shape of the image
    height, width = corrupted_image.shape

    # create an input
    input = Input(shape=(height, width, 1))

    # create a base model
    base = base_model(input)

    # Create a model form inputs
    model = Model(inputs=input, outputs=base)

    x = (corrupted_image - 0.5)[np.newaxis, :, :, np.newaxis]
    y = model.predict(x)[0, :, :, 0]

    return (0.5 + y).clip(0, 1).astype(np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    '''
    Adds gausian noise to an image
    :param image: A grayscale float64 image in range [0,1]
    :param min_sigma: The minimal variance of the gaussian distribution
    :param max_sigma: The maximal variance of the gaussian distribution
    :return: The image corrupted by a gaussian noise
    '''

    sigma = np.random.uniform(min_sigma, max_sigma)
    rand_gaussian_noise_matrix = np.random.normal(0, sigma, image.shape)

    corrupted_image = image + rand_gaussian_noise_matrix
    corrupted_image_normalized = np.divide(np.round(corrupted_image * (255)), (255))
    corrupted_image_normalized = np.clip(corrupted_image_normalized, 0, 1)

    return corrupted_image_normalized


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    '''
    Used for trainning a neural network to denoise images with iid type gausian blur
    :param num_res_blocks: The number of residual blocks in the network
    :param quick_mode: Bool values representing if trainning is done in the fast mode
    :return: The trained model
    '''

    def corruption_func(im):
        return add_gaussian_noise(im, DENOISE_MIN_SIGMA, DENOISE_MAX_SIGMA)

    # Get images to denoise
    images = sol5_utils.images_for_denoising()

    # Inits a neural network model
    model = build_nn_model(DENOISE_PATCH_SIZE, DENOISE_PATCH_SIZE, DENOISE_CHANNELS_NUM, num_res_blocks)

    # Train The model
    if quick_mode:
        train_model(model, images, corruption_func, QUICK_BATCH_NUM,
                    QUICK_STEPS_PER_EPOCH, QUICK_TOTAL_EPOCH, QUICK_VALIDATION_SET)
    else:
        train_model(model, images, corruption_func, BATCH_NUM,
                    STEPS_PER_EPOCH, DENOISE_TOTAL_EPOCH, VALIDATION_SET)
    return model


def add_motion_blur(image, kernel_size, angle):
    '''
    Adding motion blur to a given image
    :param image:Grayscale image in the [0,1] range of float64
    :param kernel_size: The size of the kernel.An ood integer
    :param angle: angle in range [0,pi)
    :return: Image blurred
    '''
    blur_kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    corrupted = convolve(image, blur_kernel)
    corrupted = np.divide(np.round(corrupted * (255)), (255))
    corrupted = np.clip(corrupted, 0, 1)
    return corrupted


def random_motion_blur(image, list_of_kernel_sizes):
    '''
    Adds random motion blur with  a random kernel size and an angle in range [0,pi)
    :param image:Grayscale image in the [0,1] range of float64
    :param list_of_kernel_sizes:  list of ood numbers that represnt kernel size
    :return: The image blurred
    '''

    random_idx = np.random.randint(len(list_of_kernel_sizes))
    rand_kernel_size = list_of_kernel_sizes[random_idx]

    rand_angle = np.random.uniform(low=MIN_RAND_ANGLE, high=MAX_RAND_ANGLE)

    return ((add_motion_blur(image, rand_kernel_size,
                             rand_angle) * 255).round() / 255).clip(0, 1)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    '''
    Training for motion deblurring on blurred images
    :param num_res_blocks: The number of res blocks in the network
    :param quick_mode: Bool values representing if trainning is done in the fast mode
    :return:  A trained model
    '''

    def corruption_func(im):
        return random_motion_blur(im, [DEBLUR_KERNEL_SIZE])

    images = sol5_utils.images_for_deblurring()
    model = build_nn_model(DEBLUR_PATCH_SIZE, DEBLUR_PATCH_SIZE, DEBLUR_CHANNELS, num_res_blocks)

    if quick_mode:
        train_model(model, images, corruption_func, QUICK_BATCH_NUM,
                    QUICK_STEPS_PER_EPOCH, QUICK_TOTAL_EPOCH, QUICK_VALIDATION_SET)

    else:
        train_model(model, images, corruption_func, BATCH_NUM,
                    STEPS_PER_EPOCH, DEBLUR_EPOCHS, VALIDATION_SET)

    return model


def deep_prior_restore_image(corrupted_image):
    '''

    :param corrupted_image:
    :return:
    '''
    pass
