import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import correlate2d

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
    """

    padded = np.pad(vals, ((int(vals.shape[0]/2), int(vals.shape[1]/2)),), mode=pad_mode)
    # padded = vals
    f = fft2(padded)
    l2norm = np.absolute(f)**2
    res = ifft2(l2norm)
    

    # return correlate2d(vals, vals, mode='same', boundary='fill')
    return np.real(res)


# def autocorrelation(x):
#     xp = ifftshift((x - np.average(x))/np.std(x))
#     n, = xp.shape
#     xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
#     f = fft(xp)
#     p = np.absolute(f)**2
#     pi = ifft(p)
#     return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)

# def autocorrelation(x) :
#     # xp = (x - np.average(x))/np.std(x)
#     f = fft2(xp)
#     p = np.absolute(f)**2
#     pi = ifft2(p)
#     return np.real(pi)[:int(xp.shape[0]/2), :int(xp.shape[1]/2)]/(len(xp))
