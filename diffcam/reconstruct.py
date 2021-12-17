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

from pycsou.linop import Gradient
from pycsou.opt import APGD
from pycsou.func import SquaredL2Loss, SquaredL2Norm, Segment, L1Norm, NonNegativeOrthant
from pycsou.linop import Convolve2D
from recon import Recon
from plot import plot_image

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




    start_time = time.time()
    # TODO : setup for your reconstruction algorithm
    # Gop is our mask (tape) described by our psf
    # Gop = Convolve2D(size=data.size,
    #                   filter=psf, shape=data.shape)
    # Gop.compute_lipschitz_cst()

    # varlambda = .005
    # loss = SquaredL2Loss(dim=data.size, data=data.flatten())

    # # we should have F = 1/2 * ‖y − Hx‖ + λ‖x‖      with ‖.‖ squared L2 norm
    # ridgeF = ((1/2) * loss * Gop) + (varlambda * SquaredL2Norm(dim=data.size))
    # # lasso, same but l1 norm (non diffirentiable)
    # lassoF = ((1/2) * loss * Gop)
    # lassoG = varlambda * L1Norm(dim=data.size)
    # # non-negative least square, same but non-negativity prior (non diffirentiable)
    # nnF = lassoF
    # nnG = NonNegativeOrthant(dim=data.size) # varlambda should have no effect in this case
    ####################
    # apgd = APGD(dim=data.size, F=ridgeF, G=None, verbose=None)  # Initialise APGD with only our functional F to minimize
    # apgd = APGD(dim=data.size, F=lassoF, G=lassoG, verbose=None)  
    # apgd = APGD(dim=data.size, F=nnF, G=nnG, verbose=None)
    solver = Recon(data, psf, mode='ridge')
    print(f"setup time : {time.time() - start_time} s")







    start_time = time.time()
    # TODO : apply your reconstruction
    allout = solver.iterate() # Run APGD
    plt.figure()
    estimate = solver.get_estimate()
    np.save('rgb_estimate.npy', estimate)
    plot_image(estimate)
    print(f"proc time : {time.time() - start_time} s")

    if not no_plot:
        plt.show()
    if save:
        print(f"Files saved to : {save}")


if __name__ == "__main__":
    reconstruction()