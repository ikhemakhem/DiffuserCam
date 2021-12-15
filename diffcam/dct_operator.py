"""
Created on Sun Dec 13 17:51:21 2021

@author: elizealwash & ludvigdillen
"""
from pycsou.core import LinearOperator
import numpy as np
from scipy.fft import dctn, idctn

class DCT(LinearOperator):
    """
    Linear operator for the DCT (Discrete Fourier Transform) and its adjoint, the IDCT
    (Inverse Discrete Fourier Transform). This implemenation supports the multidimensional
    DCT.
    """
    def __init__(self, size: int, dtype: type = np.float64):
        super(DCT, self).__init__(shape=(size, size))

    def __call__(self, x: np.ndarray, my_type = 2, my_norm = None) -> np.ndarray:
         return dctn(x, type=my_type, norm = my_norm)

    def adjoint(self, y: np.ndarray, my_type = 2, my_norm = None) -> np.ndarray:
        return idctn(y, type=my_type, norm = my_norm)



