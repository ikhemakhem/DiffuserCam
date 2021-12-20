from pycsou.core import LinearOperator
from pycsou.func import DifferentiableFunctional
import numpy as np
from scipy.fft import dctn, idctn

class DCT(LinearOperator):
    """
    Linear operator for the DCT (Discrete Fourier Transform) and its adjoint, the IDCT
    (Inverse Discrete Fourier Transform). This implemenation supports the multidimensional
    DCT.
    """
    def __call__(self, x: np.ndarray, my_type = 2, my_norm = 'ortho') -> np.ndarray:
        return dctn(x, type=my_type, norm = my_norm)

    def adjoint(self, y: np.ndarray, my_type = 2, my_norm = 'ortho') -> np.ndarray:
        return idctn(y, type=my_type, norm = my_norm)

class IDCT(LinearOperator):
    """
    Linear operator for the IDCT (Inverse Discrete Fourier Transform) and its adjoint, the DCT
    (Discrete Fourier Transform). This implemenation supports the multidimensional
    IDCT.
    """
    def __call__(self, x: np.ndarray, my_type = 2, my_norm = 'ortho') -> np.ndarray:
        return idctn(x, type=my_type, norm = my_norm)

    def adjoint(self, y: np.ndarray, my_type = 2, my_norm = 'ortho') -> np.ndarray:
        return dctn(y, type=my_type, norm = my_norm)


class HuberNorm(DifferentiableFunctional):
    """
    Constructs the Huber Norm of differentiable functions, where DifferentiableFunctions is the base class of differentiable functions.
    """
    def __init__(self, dim: int, delta: float):
        """
        Parameters
        ----------
        dim : int
            Dimension of differentiable function.
        delta : float 
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(HuberNorm, self).__init__(dim=dim, diff_lipschitz_cst=1)
        self.delta = delta
    def __call__(self, x: np.ndarray) -> float:
        """

        Parameters
        ----------
        x : np.ndarray
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        """
        z = x
        for i in range(z.size):
            if abs(z[i]) <= self.delta:
                z[i] = 0.5*z[i]*z[i]
            else:
                z[i] = self.delta*(abs(z)-self.delta/2) 
        return np.sum(z)
    
    def jacobianT(self, x: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        x : np.ndarray
            DESCRIPTION.

        Returns
        -------
        grad : np.ndarray
            DESCRIPTION.

        """
        grad = np.empty_like(x)
        for i in range(x.size):
            if abs(x[i])<= self.delta:
                grad[i] = x[i]
            elif x[i] > self.delta:
                grad[i] = self.delta
            else:
                grad[i] = self.delta
        return grad
