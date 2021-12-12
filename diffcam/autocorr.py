import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

def autocorr2d(vals, pad_mode="reflect"):
    """
    Compute 2-D autocorrelation of image via the FFT.

    Parameters
    ----------
    vals : py:class:`~numpy.ndarray`
        2-D image.
    pad_mode : str
        Desired padding. See NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Return
    ------
    autocorr : py:class:`~numpy.ndarray`

    Created on Sun Dec 12 18:13:00 2021

    @author: ludvigdillen
    """
    pad_width = int(vals.shape[0]/2)  # length of padding left/right of the 2-D array
    pad_height = int(vals.shape[1]/2)  # length of padding top/bottom of the 2-D array
    # The resulting 2-D array from this operation has twice the width and height compared
    # with the input 2-Darray.
    padded_vals = np.pad(vals, ((pad_width, pad_width),(pad_height,pad_height)), mode=pad_mode)
    f = fft2(padded_vals)
    l2norm = np.absolute(f)**2  # computing the autocorrelation in the Fourier domain
    res = ifftshift(ifft2(l2norm), axes=None)  # ifftshift centrals the signal around zero
    # Keep the samples around zero with the constraint that the output is of the same size as
    # the input and that the autocorrelation is symmetric around 0.
    res = res[pad_width:3*pad_width, pad_height:3*pad_height]
    return np.real(res)

