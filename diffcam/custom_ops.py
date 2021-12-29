import numpy as np
from pycsou.core import LinearOperator
from pycsou.func import DifferentiableFunctional
from scipy.fft import dctn, idctn


class DCT(LinearOperator):
    """
    Linear operator for the DCT (Discrete Fourier Transform) and its adjoint, the IDCT
    (Inverse Discrete Fourier Transform). This implemenation supports the multidimensional
    DCT.
    """
    def __call__(self, x: np.ndarray, type = 2, norm = 'ortho') -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            The input array.
        type : int
            Type of the DCT.
        norm : string
            Normalization mode.
        Returns
        -------
        y : np.ndarray
            The transformed input array.
        """
        return dctn(x, type=type, norm = norm)

    def adjoint(self, y: np.ndarray, my_type = 2, my_norm = 'ortho') -> np.ndarray:
        """
        Parameters
        ----------
        y : np.ndarray
            The input array.
        type : int
            Type of the IDCT.
        norm : string
            Normalization mode.
        Returns
        -------
        x : np.ndarray
            The transformed input array.
        """
        return idctn(y, type=my_type, norm = my_norm)

class IDCT(LinearOperator):
    """
    Linear operator for the IDCT (Inverse Discrete Fourier Transform) and its adjoint, the DCT
    (Discrete Fourier Transform). This implemenation supports the multidimensional
    IDCT.
    """
    def __call__(self, x: np.ndarray, type = 2, norm = 'ortho') -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            The input array.
        type : int
            Type of the IDCT.
        norm : string
            Normalization mode.
        Returns
        -------
        y : np.ndarray
            The transformed input array.
        """
        return idctn(x, type=type, norm = norm)

    def adjoint(self, y: np.ndarray, type = 2, norm = 'ortho') -> np.ndarray:
        """
        Parameters
        ----------
        y : np.ndarray
            The input array.
        type : int
            Type of the DCT.
        norm : string
            Normalization mode.
        Returns
        -------
        x : np.ndarray
            The transformed input array.
        """
        return dctn(y, type=type, norm = norm)


class HuberNorm(DifferentiableFunctional):
    """
    Constructs the Huber Norm of differentiable functions, where DifferentiableFunctions is the
    base class of differentiable functions.
    """
    def __init__(self, dim: int, delta: float):
        """
        Parameters
        ----------
        dim : int
            Dimension of differentiable function.
        delta : float
            Hyperparameter which decides if the norm resembles a l1 or l2 norm the most. The higher
            the value, the more does the norm resemble a l2 norm and vice versa.

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
            Argument which the Huber norm calculated for.

        Returns
        -------
        float
            The Huber norm.

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
            The value for which we calculate the Jacobian tranpose.

        Returns
        -------
        grad : np.ndarray
            The Jacobian tranpose (gradient)

        """
        grad = np.empty_like(x)
        for i in range(x.size):
            if abs(x[i])<= self.delta:
                grad[i] = x[i]
            elif x[i] > self.delta:
                grad[i] = self.delta
            else:
                grad[i] = -self.delta
        return grad
