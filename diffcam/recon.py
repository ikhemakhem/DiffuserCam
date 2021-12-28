import abc
import numpy as np
import pathlib as plib
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from diffcam.plot import plot_image
from pycsou.linop import Gradient
from pycsou.opt import APGD, PDS
from pycsou.func import SquaredL2Loss, SquaredL2Norm, L1Norm, NonNegativeOrthant
from diffcam.custom_ops import IDCT, HuberNorm
from diffcam.fast_convolve2D import FastConvolve2D

class ReconstructionAlgorithm(abc.ABC):
    def __init__(self, psf, dtype=np.float32):

        self._is_rgb = True if len(psf.shape) == 3 else False
        if self._is_rgb:
            self._psf = psf
            self._n_channels = 3
        else:
            self._psf = psf[:, :, np.newaxis]
            self._n_channels = 1
        self._psf_shape = np.array(self._psf.shape)

        if dtype:
            self._psf = self._psf.astype(dtype)
            self._dtype = dtype
        else:
            self._dtype = self._psf.dtype
        if self._dtype == np.float32 or dtype == "float32":
            self._complex_dtype = np.complex64
        elif self._dtype == np.float64 or dtype == "float64":
            self._complex_dtype = np.complex128
        else:
            raise ValueError(f"Unsupported dtype : {self._dtype}")

        # cropping / padding indices
        self._padded_shape = 2 * self._psf_shape[:2] - 1
        self._padded_shape = np.array([next_fast_len(i) for i in self._padded_shape])
        self._padded_shape = np.r_[self._padded_shape, [self._n_channels]]
        self._start_idx = (self._padded_shape[:2] - self._psf_shape[:2]) // 2
        self._end_idx = self._start_idx + self._psf_shape[:2]

        # pre-compute operators / outputs
        self._image_est = None
        self._data = None
        self.reset()

    @abc.abstractmethod
    def reset(self):
        return

    @abc.abstractmethod
    def _forward(self):
        return

    @abc.abstractmethod
    def _backward(self, x):
        return

    @abc.abstractmethod
    def _update(self):
        return

    @abc.abstractmethod
    def _form_image(self):
        return

    def set_data(self, data):
        if not self._is_rgb:
            assert len(data.shape) == 2
            data = data[:, :, np.newaxis]
        assert len(self._psf_shape) == len(data.shape)
        self._data = data

    def get_image_est(self):
        return self._form_image()

    def apply(
        self, n_iter=100, disp_iter=10, plot_pause=0.2, plot=True, save=False, gamma=None, ax=None
    ):
        assert self._data is not None, "Must set data with `set_data()`"

        if (plot or save) and disp_iter is not None:
            if ax is None:
                ax = plot_image(self._data, gamma=gamma)
        else:
            ax = None
            disp_iter = n_iter + 1

        for i in range(n_iter):
            self._update()

            if (plot or save) and (i + 1) % disp_iter == 0:
                ax = plot_image(self._form_image(), ax=ax, gamma=gamma)
                ax.set_title("Reconstruction after iteration {}".format(i + 1))
                if save:
                    plt.savefig(plib.Path(save) / f"{i + 1}.png")
                if plot:
                    plt.draw()
                    plt.pause(plot_pause)

        final_im = self._form_image()
        if plot or save:
            ax = plot_image(final_im, ax=ax, gamma=gamma)
            ax.set_title("Final reconstruction after {} iterations".format(n_iter))
            if save:
                plt.savefig(plib.Path(save) / f"{n_iter}.png")
            return final_im, ax
        else:
            return final_im

def get_solver(data, psf, mode, Gop, loss, lambda1=.005, huber_delta=1.5,  acceleration='CD'):
    apdg_modes = ['ridge', 'lasso', 'nn', 'dct', 'huber']
    pds_modes = ['nnL1']

    Gop.compute_lipschitz_cst()
    # Tikhonov regularization (squaredL2norm is differentiable)
    ridgeF = ((1/2) * loss * Gop) + (lambda1 * SquaredL2Norm(dim=data.size))

    # lasso regularization (same but l1 norm (non diffirentiable))
    lassoF = ((1/2) * loss * Gop)
    lassoG = lambda1 * L1Norm(dim=data.size)

    # non-negative prior (non diffirentiable)
    nnF = lassoF
    nnG = NonNegativeOrthant(dim=data.size) # lambda1 should have no effect in this case

    # Gradient oprator for TV-regularization and huber regularization
    D = Gradient(shape=data.shape)
    D.compute_lipschitz_cst()

    # Total-Variation and non-negative regularization (both proximable)
    pdsF = ((1/2) * loss * Gop)
    pdsG = NonNegativeOrthant(dim=data.size)
    pdsH = lambda1 * L1Norm(dim=D.shape[0])

    # Lasso with DCT
    idct = IDCT(shape=[data.size,data.size])
    idct.compute_lipschitz_cst()
    dctF = (1/2) * loss * Gop * idct
    dctG = lambda1 * L1Norm(dim=data.size)

    # Huber Regularization
    huberF = ((1/2) * loss * Gop) + lambda1 * HuberNorm(dim = D.shape[0], delta=huber_delta)*D
    huberG = NonNegativeOrthant(dim=data.size)

    if mode == 'ridge':
        solver = APGD(dim=data.size, F=ridgeF, G=None, verbose=None, acceleration=acceleration) 
    elif mode == 'lasso':
        solver = APGD(dim=data.size, F=lassoF, G=lassoG, verbose=None, acceleration=acceleration) 
    elif mode == 'nn':
        solver = APGD(dim=data.size, F=nnF, G=nnG, verbose=None, acceleration=acceleration)
    elif mode == 'dct':
        solver = APGD(dim=data.size, F=dctF, G=dctG, verbose=None, acceleration=acceleration)
    elif mode == pds_modes[0]: #nnL1
        solver = PDS(dim=data.size, F=pdsF, G=pdsG, H=pdsH, K=D, verbose=None)
    elif mode == 'huber':
        solver = APGD(dim=data.size, F=huberF, G=huberG, verbose=None, acceleration=acceleration)
    else:
        raise Exception(str(mode) + ' mode not found.')

    if mode == 'dct':
        solver.get_estimate = lambda : idct(solver.iterand['iterand']).reshape(data.shape)
    elif mode in apdg_modes:
        solver.get_estimate = lambda : solver.iterand['iterand'].reshape(data.shape)
    else:
        solver.get_estimate = lambda : solver.iterand['primal_variable'].reshape(data.shape)


    return solver

class Recon():
    def __init__(self, data, psf, mode, lambda1=.005, huber_delta=1.5, color=True):
        assert color 
        data = {'r': data[:,:,0], 'g': data[:,:,1], 'b': data[:,:,2]}
        psf = {'r': psf[:,:,0], 'g': psf[:,:,1], 'b': psf[:,:,2]}

        Gop = {key: FastConvolve2D(size=data[key].size, filter=psf[key], shape=data[key].shape) for key in psf}
        loss = {key: SquaredL2Loss(dim=data[key].size, data=data[key].flatten()) for key in data}

        self.solver = {key: get_solver(data[key], psf[key], mode, Gop[key], loss[key], lambda1, huber_delta) for key in data}


    def iterate(self):
        out = []
        for key in self.solver:
            out.append(self.solver[key].iterate())

        return out

    def get_estimate(self):
        estimate = np.array([self.solver[key].get_estimate() for key in self.solver])
        to_return = np.empty((estimate.shape[1], estimate.shape[2], estimate.shape[0]))
        to_return[:,:,0] = estimate[0]
        to_return[:,:,1] = estimate[1]
        to_return[:,:,2] = estimate[2]
        return to_return
