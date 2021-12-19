import abc
import numpy as np
import pathlib as plib
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from diffcam.plot import plot_image
from pycsou.linop import Gradient, Convolve2D
from pycsou.opt import APGD, PDS
from pycsou.func import SquaredL2Loss, SquaredL2Norm, Segment, L1Norm, NonNegativeOrthant


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

def get_solver(data, psf, mode, Gop, loss, varlambda=.005, acceleration='CD'):
    apdg_modes = ['ridge', 'lasso', 'nn']
    pds_modes = ['nnL1']

    Gop.compute_lipschitz_cst()
    # we should have F = 1/2 * ‖y − Hx‖ + λ‖x‖      with ‖.‖ squared L2 norm
    ridgeF = ((1/2) * loss * Gop) + (varlambda * SquaredL2Norm(dim=data.size))
    # lasso, same but l1 norm (non diffirentiable)
    lassoF = ((1/2) * loss * Gop)
    lassoG = varlambda * L1Norm(dim=data.size)
    # non-negative least square, same but non-negativity prior (non diffirentiable)
    nnF = lassoF
    nnG = NonNegativeOrthant(dim=data.size) # varlambda should have no effect in this case

    # whatever the pds one was called
    pdsD = Gradient(shape=data.shape)
    pdsD.compute_lipschitz_cst()
    mu = 0.035 * np.max(pdsD(Gop.adjoint(data.flatten()))) # Penalty strength

    pdsF = ((1/2) * loss * Gop)
    pdsG = NonNegativeOrthant(dim=data.size)
    pdsH = mu * L1Norm(dim=pdsD.shape[0])

    if mode == 'ridge':
        solver = APGD(dim=data.size, F=ridgeF, G=None, verbose=None, acceleration=acceleration)  # Initialise APGD with only our functional F to minimize
    elif mode == 'lasso':
        solver = APGD(dim=data.size, F=lassoF, G=lassoG, verbose=None, acceleration=acceleration) 
    elif mode == 'nn':
        solver = APGD(dim=data.size, F=nnF, G=nnG, verbose=None, acceleration=acceleration)
    elif mode == pds_modes[0]:
        solver = PDS(dim=data.size, F=pdsF, G=pdsG, H=pdsH, K=pdsD, verbose=None)
    else:
        raise Exception(str(mode) + ' mode not found.')


    if mode in apdg_modes:
        solver.get_estimate = lambda : solver.iterand['iterand'].reshape(data.shape)
    else:
        solver.get_estimate = lambda : solver.iterand['primal_variable'].reshape(data.shape)


    return solver

class Recon():
    def __init__(self, data, psf, mode, varlambda=.005, color=True):
        assert color #this was not a question.
        data = {'r': data[:,:,0], 'g': data[:,:,1], 'b': data[:,:,2]}
        psf = {'r': psf[:,:,0], 'g': psf[:,:,1], 'b': psf[:,:,2]}
        to_save = np.array(list(data.values()))
        print('to_save', to_save.shape)
        np.save('semi-transposed_data.npy', to_save)
        print('pls')

        Gop = {key: Convolve2D(size=data[key].size, filter=psf[key], shape=data[key].shape) for key in psf}
        loss = {key: SquaredL2Loss(dim=data[key].size, data=data[key].flatten()) for key in data}

        self.solver = {key: get_solver(data[key], psf[key], mode, Gop[key], loss[key], varlambda) for key in data}


    def iterate(self):
        out = []
        for key in self.solver:
            out.append(self.solver[key].iterate())
            np.save(key+'estimate.npy', self.solver[key].get_estimate())

        return out

    def get_estimate(self):
        estimate = np.array([self.solver[key].get_estimate() for key in self.solver])
        to_return = np.empty((estimate.shape[1], estimate.shape[2], estimate.shape[0]))
        to_return[:,:,0] = estimate[0]
        to_return[:,:,1] = estimate[1]
        to_return[:,:,2] = estimate[2]
        return to_return