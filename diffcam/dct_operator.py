"""
Created on Sun Dec 13 17:51:21 2021

@author: elizealwash & ludvigdillen
"""
from pycsou.core import LinearOperator
from numpy import np
from scipy.fft import dctn, idctn

class RepCol(LinearOperator):
    """
    Linear operator for the DCT (Discrete Fourier Transform) and its adjoint, the IDCT
    (Inverse Discrete Fourier Transform). This implemenation supports the multidimensional
    DCT.
    """
    def __init__(self, size: int, reps: int, dtype: type = np.float64):
        self.reps = reps
        super(RepCol, self).__init__(shape=(size*reps, size))

    def __call__(self, x: np.ndarray, my_type=2, my_s=None, my_axes=None, my_norm=None, \
        my_overwrite_x=False, my_workers=None) -> np.ndarray:
        return dctn(x, type=my_type, s=my_s, axes=my_axes, norm=my_norm, \
            overwrite_x=my_overwrite_x, workers=my_workers)

    def adjoint(self, x: np.ndarray, my_type=2, my_s=None, my_axes=None, my_norm=None, \
        my_overwrite_x=False, my_workers=None) -> np.ndarray:
        return idctn(x, type=my_type, s=my_s, axes=my_axes, norm=my_norm, \
            overwrite_x=my_overwrite_x, workers=my_workers)
