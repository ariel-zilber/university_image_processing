from math import floor, ceil
import numpy as np
import scipy.io.wavfile as waveFile
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from skimage.color import rgb2gray
from imageio import imread

## names of output files
CHANGED_RATE_FILE_NAME = "change_rate.wav"
CHANGED_SAMPLE_FILE_NAME = "change_samples.wav"

# possible reps:
GRAY_SCALE = 1
RGB = 2


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


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


# 1.1 1D DFT
def DFT(signal):
    '''
    Transforms a 1D discrete signal to its Fourier representation
    :param signal:an array of dtype float64 with shape (N,1)
    :return: complex Fourier signal:an array of dtype complex128 with the same shape
    '''

    # The length of the signal:
    N = signal.size

    # The range to work on
    u = np.arange(N)
    x = u.reshape((N, 1))

    # An exponent matrix to multiple the signal one of size(N,N)
    e = np.exp(-2j * np.pi * u * x / float(N))

    # perform dot operation  (N,N) x (N,1)
    # The result is of the size (N,1)

    return np.dot(e, signal)


def IDFT(fourier_signal):
    '''
    Inverse  Fourier transform of of a 1d discrete signal
    :param  fourier_signal:an array of dtype complex128 with shape (N,1)
    :return: complex signal:an array of dtype float64 with shape (N,1)
    '''

    # The length of the signal:
    N = fourier_signal.size

    # The range to work on
    u = np.arange(N)
    x = u.reshape((N, 1))

    # An exponent matrix to multiple the signal one of size(N,N)
    e = np.exp(2j * np.pi * u * x / float(N))

    # perform dot operation  (N,N) x (N,1)
    # The result is of the size (N,1)
    f = (1 / float(N)) * np.dot(e, fourier_signal)

    return f


# 1.2 2D DFT

def DFT2(image):
    '''
    convert a 2D discrete signal to its Fourier representation

    :param image: a grayscale image of dtype float64
    :return:  a 2D array of dtype complex128
    '''

    fourier_image = np.zeros(image.shape).astype(np.complex128)

    for col in range(fourier_image.shape[1]):
        fourier_image[:, col] = DFT(image[:, col])

    for row in range(fourier_image.shape[0]):
        fourier_image[row, :] = DFT(fourier_image[row, :])

    return fourier_image


def IDFT2(fourier_image):
    '''
    reverse Fourier representation to a 2D discrete signal
    :param fourier_image: a 2D array of dtype complex128
    :return: a grayscale image of dtype float64
    '''

    image = np.zeros(fourier_image.shape).astype(np.complex128)

    for row in range(image.shape[0]):
        image[row, :] = IDFT(fourier_image[row, :])

    for col in range(image.shape[1]):
        image[:, col] = IDFT(image[:, col])

    return image.astype(np.complex128)


# 2.1 Fast forward by rate change
def change_rate(filename, ratio):
    '''
    changes the duration of an audio file by keeping the same samples, but changing the
    sample rate written in the file header.
    Given a WAV file, this function saves the audio
    in a new file called "change_rate.wav."
    :param filename: the path of a wav file to change
    :param ratio: the new ratio
    '''
    rate, data = waveFile.read(filename)
    new_rate = rate * ratio

    waveFile.write(CHANGED_RATE_FILE_NAME, int(new_rate), data)


# 2.2 Fast forward using Fourier
def change_samples(filename, ratio):
    '''
     function that changes the duration of an audio file by reducing the number of samples
     using Fourier. This function does not change the sample rate of the given file.
    :param filename: a string representing the path to a WAV file
    :param ratio: positive float64 representing the duration change
    :return:
    '''
    rate, data = waveFile.read(filename)
    if len(data.shape) == 1:
        mono_channel = data
    else:
        mono_channel = data[:, 0]
    result = resize(mono_channel, ratio)

    result=result.astype(np.float64)
    result=normalize(result)
    waveFile.write(CHANGED_SAMPLE_FILE_NAME, rate, result)


def normalize(data):
    data = np.real(data).astype(np.float64)

    max_value = data[0]

    for i in range(0, data.shape[0]):
        if (data[i] > max_value):
            max_value = data[i]

    return data / max_value


def resize(data, ratio):
    '''

    :param data:a 1D ndarray of dtype float64 representing the sample points
    :param ratio: the new  ratio
    :return: the file with the new ratio
    '''

    fourier_data = DFT(data)
    fourier_data_shifted = np.fft.fftshift(fourier_data)

    n = fourier_data_shifted.shape[0]
    new_size = int(floor(n / ratio))

    if (ratio > 1):
        samples_to_delete = n - new_size
        start_index = samples_to_delete // 2
        end_index = n - start_index


        if (ceil(n / ratio) % 2 != 0):
            end_index -= 1

        changed_data_freq = fourier_data_shifted[start_index: end_index]

    else:
        samples_to_pad = new_size - n
        pad_at_start = np.zeros(samples_to_pad // 2)

        if (ceil(n / ratio) % 2 != 0):
            pad_at_end = np.zeros((samples_to_pad // 2)+1)
        else:
            pad_at_end = np.zeros(samples_to_pad // 2)


        changed_data_freq = np.hstack((pad_at_start, fourier_data_shifted, pad_at_end))

        # shift back
        changed_data_freq = np.fft.ifftshift(changed_data_freq)

    changed_data = IDFT(changed_data_freq)

    return changed_data.astype(np.complex128)


# 2.3 Fast forward using Spectrogram
def resize_spectrogram(data, ratio):
    '''

    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: the resized signal
    '''

    fourier_data_rows = stft(data)
    fourier_data_rows_resized = []

    for i in range(0, len(fourier_data_rows)):
        fourier_data_rows_resized.append(resize(fourier_data_rows[i], ratio))

    result = np.asarray(fourier_data_rows_resized)

    resized_data = istft(result)

    return resized_data


#  2.4 Fast forward using Spectrogram and phase vocoder
def resize_vocoder(data, ratio):
    spec = stft(data)

    spec_phased = phase_vocoder(spec, ratio)
    result = istft(spec_phased)
    return result


# 3 Image derivatives
# 3.1 Image derivatives in image space
def conv_der(im):
    '''
    computes the magnitude of image derivatives

    :param im: grayscale images of type float64
    :return: output is grayscale images of type float64 and the  magnitude
        of the derivative, with the same dtype and shape
    '''

    # get the image derivates
    dX = np.array([[0.5, 0, -0.5]], dtype=np.float64)
    dY = dX.transpose()

    dx = signal.convolve2d(im, dX, mode='same')
    dy = signal.convolve2d(im, dY, mode='same')

    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


# 3.2 Image derivatives in Fourier space
def fourier_der(im):
    '''
    computes the magnitude of image derivatives using Fourier transform.
    :param im: float64 grayscale images.
    :return: float64 grayscale images.

    '''

    # calculate the fourier transform of the image
    fourier_trans = DFT2(im)

    # shift the image
    shifted_im = np.fft.fftshift(fourier_trans)

    N = im.shape[0]
    M = im.shape[1]
    u_array = np.arange(-N // 2, N // 2)
    v_array = np.arange(-M // 2, M // 2)

    # multiple each f(u,v) by U
    result1 = np.transpose(np.transpose(shifted_im) * u_array)

    # perform inverse  fourier transform
    inverse_dx = IDFT2(result1)

    # multiple each f(u,v) by v
    result2 = shifted_im * v_array

    # perform inverse  fourier transform
    inverse_dy = IDFT2(result2)

    # calculate the magnitude
    magnitude = np.sqrt(np.abs(inverse_dx) ** 2 + np.abs(inverse_dy) ** 2)

    return magnitude
