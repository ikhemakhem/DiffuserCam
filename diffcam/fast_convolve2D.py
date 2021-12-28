import numpy as np
from pycsou.core import LinearOperator
from pycsou.linop.base import PyLopLinearOperator
from pylops import LinearOperator
from scipy.fft import ifftshift, irfft2, rfft2


class pylopsFastConvolve2D(LinearOperator):
    def __init__(self, N, h, dims, dtype='float64'):
        """
        Parameters
        ----------
        N : int
            Number of elements of 2D array to convolve with.
        h : np.ndarray
            Filter to convolve with.
        dims : tuple
            Shape of elements of 2D array to convolve with.
        dtype : dtype, optional
            Datatype of convolution operator. The default is 'float64'.

        Raises
        ------
        ValueError
            Product of dims must equal N!

        Returns
        -------
        None.

        """
        self.h = h
        if np.prod(dims) != N:
            raise ValueError('product of dims must equal N!')
        else:
            self.dims = np.array(dims)
        self.shape = (np.prod(self.dims), np.prod(self.dims))
        self.dtype = np.dtype(dtype)
        self.explicit = False
        # extra for the fft_filter
        self.pad_width = int(h.shape[0]/2)  # length of padding left/right of the 2-D array
        self.pad_height = int(h.shape[1]/2)  # length of padding top/bottom of the 2-D array
        padded_h = np.pad(h, ((self.pad_width, self.pad_width),(self.pad_height,self.pad_height)))
        self.fft = rfft2(padded_h, axes=(0, 1))
        self.pad_matrix = np.zeros(shape=padded_h.shape, dtype=float)
        

    def _matvec(self, x):
        """
        Parameters
        ----------
        x : np.ndarray
            Flattened data to convolve with.

        Returns
        -------
        y : np.ndarray
            Flattened output from convolution.

        """
        x = np.reshape(x, self.dims)
        padded_x = self.pad_matrix
        padded_x[self.pad_width:3*self.pad_width, self.pad_height:3*self.pad_height] = x
        y = ifftshift(irfft2(self.fft * rfft2(padded_x, axes=(0, 1)), axes=(0, 1),),axes=(0, 1),)
        y = y[self.pad_width:3*self.pad_width, self.pad_height:3*self.pad_height]
        y = y.ravel()
        return y


    def _rmatvec(self, x):
        """
        Parameters
        ----------
        x : np.ndarray
            Flattened data to correlate with.

        Returns
        -------
        y : np.ndarray
            Flattened output from correlation.

        """
        x = np.reshape(x, self.dims)
        padded_x = self.pad_matrix
        padded_x[self.pad_width:3*self.pad_width, self.pad_height:3*self.pad_height] = x
        y = ifftshift(irfft2(np.conj(self.fft) * rfft2(padded_x, axes=(0, 1)), axes=(0, 1)),axes=(0, 1),)
        y = y[self.pad_width:3*self.pad_width, self.pad_height:3*self.pad_height]
        y = y.ravel()
        return y


def FastConvolve2D(size: int, filter: np.ndarray, shape: tuple) -> PyLopLinearOperator:
    """
    Parameters
    ----------
    size : int
        Number of elements of 2D array to convolve with.
    filter : np.ndarray
        Filter to convolve with.
    shape : tuple
        Shape of elements of 2D array to convolve with.
    Returns
    -------
    PyLopLinearOperator
        Constructs a linear operator from a :py:class:`pylops.LinearOperator` instance.
    """
    PyLop = pylopsFastConvolve2D(N=size, h=filter, dims=shape)
    return PyLopLinearOperator(PyLop)
