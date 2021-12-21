"""
This script will load the PSF data and raw measurement for the reconstruction
that can implement afterwards.
```bash
python scripts/reconstruction_template.py --psf_fp data/psf/diffcam_rgb.png \
--data_fp data/raw_data/thumbs_up_rgb.png
```
"""

import os
import time
import pathlib as plib
import click
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from diffcam.io import load_data

from scipy.fft import rfft2, irfft2, ifftshift 

from pycsou.core import LinearOperator
from pycsou.opt import APGD
from pycsou.func import SquaredL2Loss, L1Norm
from pycsou.linop.base import PyLopLinearOperator
from pylops import LinearOperator


@click.command()
@click.option(
    "--psf_fp",
    type=str,
    help="File name for recorded PSF.",
)
@click.option(
    "--data_fp",
    type=str,
    help="File name for raw measurement data.",
)
@click.option(
    "--n_iter",
    type=int,
    default=500,
    help="Number of iterations.",
)
@click.option(
    "--downsample",
    type=float,
    default=4,
    help="Downsampling factor.",
)
@click.option(
    "--disp",
    default=50,
    type=int,
    help="How many iterations to wait for intermediate plot/results. Set to negative value for no intermediate plots.",
)
@click.option(
    "--flip",
    is_flag=True,
    help="Whether to flip image.",
)
@click.option(
    "--save",
    is_flag=True,
    help="Whether to save intermediate and final reconstructions.",
)
@click.option(
    "--gray",
    is_flag=True,
    help="Whether to perform construction with grayscale.",
)
@click.option(
    "--bayer",
    is_flag=True,
    help="Whether image is raw bayer data.",
)
@click.option(
    "--no_plot",
    is_flag=True,
    help="Whether to no plot.",
)
@click.option(
    "--bg",
    type=float,
    help="Blue gain.",
)
@click.option(
    "--rg",
    type=float,
    help="Red gain.",
)
@click.option(
    "--gamma",
    default=None,
    type=float,
    help="Gamma factor for plotting.",
)
@click.option(
    "--single_psf",
    is_flag=True,
    help="Same PSF for all channels (sum) or unique PSF for RGB.",
)
def reconstruction(
    psf_fp,
    data_fp,
    n_iter,
    downsample,
    disp,
    flip,
    gray,
    bayer,
    bg,
    rg,
    gamma,
    save,
    no_plot,
    single_psf,
):
    """
    Reconstructs image using one of the regularisations; Tikhonov (ridge), LASSO or Non-negative
    
    Parameters
    ----------  
        psf_fp : np.array
            2D image.
        data_fp : np.array
            2D image
        n_iter : float
            Number of iterations of reconstruction.
        downsample : float
            Factor of which image gets downsampled.
        disp : 
        flip :
        gray :
        bayer :
        bg : float
            Blue gain.
        rg : float 
            Red gain.
        gamma : float
            factor for postprocessing. 
        save : function
            saves result automatically.
        no_plot :
        single_psf :
    
    Return
    ----------  
    Returns the reconstructed image

    """
    psf, data = load_data(
        psf_fp=psf_fp,
        data_fp=data_fp,
        downsample=downsample,
        bayer=bayer,
        blue_gain=bg,
        red_gain=rg,
        plot=not no_plot,
        flip=flip,
        gamma=gamma,
        gray=gray,
        single_psf=single_psf,
    )

    if disp < 0:
        disp = None
    if save:
        save = os.path.basename(data_fp).split(".")[0]
        timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
        save = "YOUR_RECONSTRUCTION_" + save + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)

    # varlambda = 0.00001
    # print("Now we start")
    # start_time = time.time()
    # Gop = FastConvolve2D(size=data.size, filter=psf, shape=data.shape)
    # # an example of implementation using the FastConvolve2D
    # Gop.compute_lipschitz_cst()
    # loss = SquaredL2Loss(dim=data.size, data=data.flatten())
    # idct = reconstruct_dct.IDCT(shape=[data.size,data.size])
    # idct.compute_lipschitz_cst()
    # F_func = (1/2) * loss * Gop * idct
    # lassoG = varlambda * L1Norm(dim=data.size)
    # apgd = APGD(dim=data.size, F=F_func, G=lassoG, verbose=None, max_iter=10)
    
    # allout = apgd.iterate() # Run APGD
    # out, _, _ = allout
    # plt.figure()
    # print('out',type(out['iterand']))
    # estimate = idct(out['iterand']).reshape(data.shape)
    # print('estimate',estimate.shape)
    # plt.imshow(estimate)
    # print(f"proc time : {time.time() - start_time} s")
    # if not no_plot:
    #     plt.show()
    # if save:
    #     print(f"Files saved to : {save}")
    # print(f"Time for FastConvolve2D: {time.time() - start_time} s")

class pylopsFastConvolveND(LinearOperator):
    def __init__(self, N, h, dims, dtype='float64'):
        """
        Parameters
        ----------
        N : TYPE
            DESCRIPTION.
        h : TYPE
            DESCRIPTION.
        dims : TYPE
            DESCRIPTION.
        dtype : TYPE, optional
            DESCRIPTION. The default is 'float64'.

        Raises
        ------
        ValueError
            DESCRIPTION.

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
        pad_width = int(h.shape[0]/2)  # length of padding left/right of the 2-D array
        pad_height = int(h.shape[1]/2)  # length of padding top/bottom of the 2-D array
        padded_h = np.pad(h, ((pad_width, pad_width),(pad_height,pad_height)))
        self.fft = rfft2(padded_h, axes=(0, 1))
        self.pad_matrix = np.zeros(shape=padded_h.shape, dtype=float)
        

    def _matvec(self, x):
        """
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        y : TYPE
            DESCRIPTION.

        """
        x = np.reshape(x, self.dims)
        padded_x = self.pad_matrix
        width_pad = int(x.shape[0]/2)
        height_pad = int(x.shape[1]/2)
        padded_x[width_pad:3*width_pad, height_pad:3*height_pad] = x
        y = ifftshift(irfft2(self.fft * rfft2(padded_x, axes=(0, 1)), axes=(0, 1),),axes=(0, 1),)
        y = y[width_pad:3*width_pad, height_pad:3*height_pad]
        y = y.ravel()
        return y


    def _rmatvec(self, x):
        """
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        y : TYPE
            DESCRIPTION.

        """
        x = np.reshape(x, self.dims)
        padded_x = self.pad_matrix
        width_pad = int(x.shape[0]/2)
        height_pad = int(x.shape[1]/2)
        padded_x[width_pad:3*width_pad, height_pad:3*height_pad] = x
        y = ifftshift(irfft2(np.conj(self.fft) * rfft2(padded_x, axes=(0, 1)), axes=(0, 1)),axes=(0, 1),)
        y = y[width_pad:3*width_pad, height_pad:3*height_pad]
        y = y.ravel()
        return y


def FastConvolve2D(size: int, filter: np.ndarray, shape: tuple) -> PyLopLinearOperator:
    """
    Parameters
    ----------
    size : int
        DESCRIPTION.
    filter : np.ndarray
        DESCRIPTION.
    shape : tuple
        DESCRIPTION.
    Returns
    -------
    PyLopLinearOperator
        Constructs a linear operator from a :py:class:`pylops.LinearOperator` instance.
    """
    PyLop = pylopsFastConvolveND(N=size, h=filter, dims=shape)
    return PyLopLinearOperator(PyLop)

    
if __name__ == "__main__":
    reconstruction()