import os
import time
import pathlib as plib

from numpy.core.numeric import identity
import click
import matplotlib.pyplot as plt
from datetime import datetime
from diffcam.io import load_data
from diffcam.recon import ReconstructionAlgorithm
from scipy import fft

from pycsou.opt import APGD, PrimalDualSplitting
from pycsou.func import SquaredL2Loss, SquaredL2Norm, Segment, L1Norm, NonNegativeOrthant, DifferentiableFunctional
from pycsou.linop import Convolve2D, Gradient, IdentityOperator
import numpy as np
from diffcam.reconstruct_huber import HuberNorm

class ReconstructionHuber(ReconstructionAlgorithm):
    def reconstruction(
        self,
        n_iter = 100
    ):
        assert self._data is not None, "Must set data with `set_data()`"
        psf = self._psf
        data = self._data
        
        start_time = time.time()
        # TODO : setup for your reconstruction algorithm
       
        lambda1 = 1
        lambda2 = 10
        print('lambda2: ', lambda2)
         # Gop is our mask (tape) described by our psf
        Gop = Convolve2D(size=data.size, filter=psf, shape=data.shape)
        Gop.compute_lipschitz_cst()
        loss = SquaredL2Loss(dim=data.size, data=data.flatten())
        F = ((1/2) * loss * Gop)

        G = lambda1*NonNegativeOrthant(dim=data.size)
        D = Gradient(shape=data.flatten().shape)
        D.compute_lipschitz_cst()

        huber_delta = 1.5
        H = lambda2 * HuberNorm(dim = data.size, delta = huber_delta)*D
        #penalty = lambda2*H_norm(D)

        K = IdentityOperator(data.size)

        print('data', data.shape)
        print('Gop', Gop.shape)
        print('D', D.shape)
        print('F', F.shape)
        print('G', G.shape)
        print('H', H.shape)

        #pds = PDS(dim=data.size, F=F, G=G, H=H, K=D, verbose=None)
        pds = PrimalDualSplitting(dim=Gop.shape[1], F=F+H, G=0.5*G, H=0.5*G, K=K, verbose=None)  # Initialise PDS
        print(f"setup time : {time.time() - start_time} s")

        start_time = time.time()
        out, _, _ = pds.iterate(n_iter)
        return out

    def _form_image(self):
        image = self._crop(self._image_est)
        image[image < 0] = 0
        return image
    
    def _forward(self):
        """Convolution with frequency response."""
        return fft.ifftshift(
            fft.irfft2(
                fft.rfft2(self._image_est, axes=(0, 1)) * self._H,
                axes=(0, 1),
            ),
            axes=(0, 1),
        )

    def _backward(self, x):
        """adjoint of forward / convolution"""
        return fft.ifftshift(
            fft.irfft2(fft.rfft2(x, axes=(0, 1)) * np.conj(self._H), axes=(0, 1)),
            axes=(0, 1),
        )

    def _crop(self, x):
        return x[self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]]
    
    def _update(self):
        pass

    def reset(self):
        pass