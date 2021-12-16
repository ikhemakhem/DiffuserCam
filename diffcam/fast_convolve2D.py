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

from numpy.core.arrayprint import format_float_positional
from diffcam.io import load_data

from pycsou.linop import Gradient
from pycsou.opt import APGD
from pycsou.func import SquaredL2Loss, SquaredL2Norm, Segment, L1Norm, NonNegativeOrthant
from pycsou.linop import Convolve2D
from pycsou.linop.base import PyLopLinearOperator
import pylops.signalprocessing as pyconv
from pylops import LinearOperator
from pylops.utils.backend import get_array_module, get_convolve, \
    get_correlate, to_cupy_conditional


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

    class pylopsFastConvolveND(LinearOperator):
        def __init__(self, N, h, dims, offset=None, dirs=None,
                    method='fft', dtype='float64'):
            ncp = get_array_module(h)
            self.h = h
            self.nh = np.array(self.h.shape)
            self.dirs = np.arange(len(dims)) if dirs is None else np.array(dirs)

            # padding
            if offset is None:
                offset = np.zeros(self.h.ndim, dtype=int)
            else:
                offset = np.array(offset, dtype=int)
            self.offset = 2 * (self.nh // 2 - offset)
            pad = [(0, 0) for _ in range(self.h.ndim)]
            dopad = False
            for inh, nh in enumerate(self.nh):
                if nh % 2 == 0:
                    self.offset[inh] -= 1
                if self.offset[inh] != 0:
                    pad[inh] = [self.offset[inh] if self.offset[inh] > 0 else 0,
                                -self.offset[inh] if self.offset[inh] < 0 else 0]
                    dopad = True
            if dopad:
                self.h = ncp.pad(self.h, pad, mode='constant')
            self.nh = self.h.shape

            # find out which directions are used for convolution and define offsets
            if len(dims) != len(self.nh):
                dimsh = np.ones(len(dims), dtype=int)
                for idir, dir in enumerate(self.dirs):
                    dimsh[dir] = self.nh[idir]
                self.h = self.h.reshape(dimsh)

            if np.prod(dims) != N:
                raise ValueError('product of dims must equal N!')
            else:
                self.dims = np.array(dims)
                self.reshape = True

            # convolve and correate functions
            self.convolve = get_convolve(h)
            self.correlate = get_correlate(h)
            self.method = method

            self.shape = (np.prod(self.dims), np.prod(self.dims))
            self.dtype = np.dtype(dtype)
            self.explicit = False

        def _matvec(self, x):
            # correct type of h if different from x and choose methods accordingly
            if type(self.h) != type(x):
                self.h = to_cupy_conditional(x, self.h)
                self.convolve = get_convolve(self.h)
                self.correlate = get_correlate(self.h)
            x = np.reshape(x, self.dims)
            y = self.convolve(x, self.h, mode='same', method=self.method)
            y = y.ravel()
            return y

        def _rmatvec(self, x):
            # correct type of h if different from x and choose methods accordingly
            if type(self.h) != type(x):
                self.h = to_cupy_conditional(x, self.h)
                self.convolve = get_convolve(self.h)
                self.correlate = get_correlate(self.h)
            x = np.reshape(x, self.dims)
            y = self.correlate(x, self.h, mode='same', method=self.method)
            y = y.ravel()
            return y
    
    def pylopsConvolve2D(N, h, dims, offset=(0, 0), nodir=None, dtype='float64',
               method='fft'):
        if h.ndim != 2:
            raise ValueError('h must be 2-dimensional')
        if nodir is None:
            dirs = (0, 1)
        elif nodir == 0:
            dirs = (1, 2)
        elif nodir == 1:
            dirs = (0, 2)
        else:
            dirs = (0, 1)

        cop = pylopsFastConvolveND(N, h, dims, offset=offset, dirs=dirs, method=method,
                                   dtype=dtype)
        return cop

    def FastConvolve2D(size: int, filter: np.ndarray, shape: tuple, fft_filter: np.ndarray,
                       dtype: type = 'float64', method: str = 'fft') -> PyLopLinearOperator:
        if (filter.shape[0] % 2) == 0:
            offset0 = filter.shape[0] // 2 - 1
        else:
            offset0 = filter.shape[0] // 2
        if (filter.shape[1] % 2) == 0:
            offset1 = filter.shape[1] // 2 - 1
        else:
            offset1 = filter.shape[1] // 2
        offset = (offset0, offset1)
        PyLop = pylopsConvolve2D(N=size, h=filter, dims=shape, nodir=None, dtype=dtype, method=method, offset=offset)
        return PyLopLinearOperator(PyLop)



    start_time = time.time()
    # TODO : setup for your reconstruction algorithm
    # Gop is our mask (tape) described by our psf
    Gop = Convolve2D(size=data.size,
                     filter=psf, shape=data.shape)
    Gop = Convolve2D(size=data.size,
                     filter=psf, shape=data.shape)
    Gop = Convolve2D(size=data.size,
                     filter=psf, shape=data.shape)
    print(f"Time for Convolve2D : {time.time() - start_time} s")
    fft_psf = np.empty((0, 0))
    start_time = time.time()

    Gop = FastConvolve2D(size=data.size, filter=psf,
                         shape=data.shape, fft_filter = fft_psf)
    Gop = FastConvolve2D(size=data.size, filter=psf,
                         shape=data.shape, fft_filter = fft_psf)
    Gop = FastConvolve2D(size=data.size, filter=psf,
                         shape=data.shape, fft_filter = fft_psf)

    print(f"Time for FastConvolve2D: {time.time() - start_time} s")




if __name__ == "__main__":
    reconstruction()