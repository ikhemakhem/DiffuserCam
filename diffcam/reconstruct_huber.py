"""
This script will load the PSF data and raw measurement for the reconstruction
that can implement afterwards.
```bash
python diffcam/reconstruct_huber.py --psf_fp data/psf/diffcam_rgb.png --data_fp data/raw_data/thumbs_up_rgb.png --gray
```
"""

import os
import time
import pathlib as plib

from numpy.core.numeric import identity
import click
import matplotlib.pyplot as plt
from datetime import datetime
from diffcam.io import load_data


from pycsou.opt import APGD, PrimalDualSplitting
from pycsou.func import SquaredL2Loss, SquaredL2Norm, Segment, L1Norm, NonNegativeOrthant, DifferentiableFunctional
from pycsou.linop import Convolve2D, Gradient, IdentityOperator
import numpy as np

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
class HuberNorm(DifferentiableFunctional):
    def __init__(self, dim: int, delta: float):
        super(HuberNorm, self).__init__(dim=dim, diff_lipschitz_cst=1)
        self.delta = delta
    def __call__(self, x: np.ndarray) -> float:
        z = x
        for i in range(z.size):
            if abs(z[i]) <= self.delta:
                z[i] = 0.5*z[i]*z[i]
            else:
                z[i] = self.delta*(abs(z)-self.delta/2) 
        return np.sum(z)
    
    def jacobianT(self, x: np.ndarray) -> np.ndarray:
        grad = np.empty_like(x)
        for i in range(x.size):
            if abs(x[i])<= self.delta:
                grad[i] = x[i]
            elif x[i] > self.delta:
                grad[i] = self.delta
            else:
                grad[i] = self.delta
        return grad

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
    print(psf.shape)
    print(data.size)

    lambda1 = 1
    lambda2 = 10
    print('lambda2: ', lambda2)
    Gop = Convolve2D(size=data.size, filter=psf, shape=data.shape)
    Gop.compute_lipschitz_cst()
    loss = SquaredL2Loss(dim=data.size, data=data.flatten())
    F = ((1/2) * loss * Gop)

    G = lambda1*NonNegativeOrthant(dim=data.size)
    #D = lambda1*FirstDerivative(size=data.size, kind='forward')
    D = Gradient(shape=data.flatten().shape)
    D.compute_lipschitz_cst()

    huber_delta = 1.5
    #huber = HuberNorm(dim = 1, delta = huber_delta)
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

    ##
    # Gop = Convolve2D(size=data.size,
    #                   filter=psf, shape=data.shape)
    # Gop.compute_lipschitz_cst()
    # loss = SquaredL2Loss(dim=data.size, data=data.flatten())
    # D = Gradient(shape=data.shape) #FirstDerivative?
    # D.compute_lipschitz_cst()
    # mu = 0.035 * np.max(D(Gop.adjoint(data.flatten()))) # Penalty strength

    # F = ((1/2) * loss * Gop)
    # G = NonNegativeOrthant(dim=data.size)
    # H = mu * L1Norm(dim=D.shape[0])

    # print('data', data.shape)
    # print('Gop', Gop.shape)
    # print('D', D.shape)
    # print('F', F.shape)
    # print('G', G.shape)
    # print('H', H.shape)
    
    # pds = PDS(dim=data.size, F=F, G=G, H=H, K=D, verbose=None)  # Initialise PDS
    print(f"setup time : {time.time() - start_time} s")




    start_time = time.time()
    # TODO : apply your reconstruction
    allout = pds.iterate() # Run APGD
    out, _, _ = allout
    plt.figure()
    estimate = out['primal_variable'].reshape(data.shape)
    plt.imshow(estimate)
    print(f"proc time : {time.time() - start_time} s")

    if not no_plot:
        plt.show()
    if save:
        print(f"Files saved to : {save}")


if __name__ == "__main__":
    reconstruction()