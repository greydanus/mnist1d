# The MNIST-1D dataset | 2024
# Sam Greydanus, Peter Steinbach

import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


# transformations of the templates which will make them harder to classify
def pad(x, padding: tuple):
    """pad signal x with random number of zeros. Note, the signal is only padded at indices given by the interval in padding

    Parameters
    ----------
    x : _type_
        signal
    padding : tuple
        (low, high) corresponds to (start,end) of padding

    Returns
    -------
    _type_
        a padded signal
    """
    low, high = padding
    p = low + int(np.random.rand() * (high - low + 1))
    if len(x.shape) == 1:
        return np.concatenate([x, np.zeros((p))])
    else:
        padding = np.zeros((x.shape[0], p))
        return np.concatenate([x, padding], axis=-1)


def shear(x, scale=10):
    # TODO: add docstring
    coeff = scale * (np.random.rand() - 0.5)
    return x - coeff * np.linspace(-0.5, 0.5, len(x))


def translate(x, max_translation):
    # TODO: add docstring
    k = np.random.choice(max_translation)
    return np.concatenate([x[-k:], x[:-k]])


def corr_noise_like(x, scale):
    # TODO: add docstring
    noise = scale * np.random.randn(*x.shape)
    return gaussian_filter(noise, 2)


def iid_noise_like(x, scale):
    # TODO: add docstring
    noise = scale * np.random.randn(*x.shape)
    return noise


def interpolate(x, N):
    # TODO: add docstring
    scale = np.linspace(0, 1, len(x))
    new_scale = np.linspace(0, 1, N)
    new_x = interp1d(scale, x, axis=0, kind="linear")(new_scale)
    return new_x


def transform(x, y, args, eps=1e-8):
    new_x = pad(x + eps, args.padding)  # pad
    new_x = interpolate(new_x, args.template_len + args.padding[-1])  # dilate
    new_y = interpolate(y, args.template_len + args.padding[-1])
    new_x *= 1 + args.scale_coeff * (np.random.rand() - 0.5)  # scale
    new_x = translate(new_x, args.max_translation)  # translate

    # add noise
    mask = new_x != 0
    new_x = mask * new_x + (1 - mask) * corr_noise_like(new_x, args.corr_noise_scale)
    new_x = new_x + iid_noise_like(new_x, args.iid_noise_scale)

    # shear and interpolate
    new_x = shear(new_x, args.shear_scale)
    new_x = interpolate(new_x, args.final_seq_length)  # subsample
    new_y = interpolate(new_y, args.final_seq_length)
    return new_x, new_y
