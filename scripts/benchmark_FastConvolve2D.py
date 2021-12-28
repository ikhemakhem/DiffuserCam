"""
Download data from here: https://drive.switch.ch/index.php/s/vmAZzryGI8U8rcE
Or full dataset here: https://github.com/Waller-Lab/LenslessLearning
```
python scripts/evaluate_mirflickr_admm.py \
--data DiffuserCam_Mirflickr_200_3011302021_11h43_seed11 \
--n_files 10 --save
```
"""

import glob
import os
import time
from datetime import datetime

import click
import numpy as np
from diffcam.io import load_psf


@click.command()
@click.option(
    "--data",
    type=str,
    help="Dataset to work on.",
)
@click.option(
    "--n_files",
    type=int,
    default=None,
    help="Number of files to apply reconstruction on. Default is apply on all.",
)
@click.option(
    "--n_iter",
    type=int,
    default=100,
    help="Number of iterations.",
)
@click.option(
    "--single_psf",
    is_flag=True,
    help="Whether to take PSF as sum of RGB channels.",
)
@click.option(
    "--save",
    is_flag=True,
    help="Whether to save reconstructions.",
)
def mirflickr_dataset(data, n_files, n_iter, single_psf, save):
    assert data is not None

    dataset_dir = os.path.join(data, "dataset")
    if os.path.isdir(dataset_dir):
        diffuser_dir = os.path.join(dataset_dir, "diffuser_images")
    else:
        diffuser_dir = os.path.join(data, "diffuser")
    psf_fp = os.path.join(data, "psf.tiff")
    downsample = 4  # has to be this for collected data!

    # determine files
    files = glob.glob(diffuser_dir + "/*.npy")
    if n_files:
        files = files[:n_files]
    files = [os.path.basename(fn) for fn in files]
    print("Number of files : ", len(files))

    psf_float, background = load_psf(
        psf_fp,
        downsample=downsample,
        return_float=True,
        return_bg=True,
        bg_pix=(0, 15),
        single_psf=single_psf,
    )

   
    modes = ['nnL1', 'ridge']
    lambdas = [1e-6, 1e-3]
    txtfile = 'benchmark_FastConvolve2D.txt'
    """ Ludvig's command to run (stand in DiffuserCam when you run)
    python scripts/evaluate_mirflickr_all.py --data DiffuserCam_Mirflickr_200_3011302021_11h43_seed11/DiffuserCam_Mirflickr_200_3011302021_11h43_seed11
    python scripts/benchmark_FastConvolve2D.py --data subset_mir_flickr_dataset/other
    """
    ################################################
    total_time = 0
    loops = len(modes)*len(lambdas)*len(files)
    print("Number of iterations: ", loops)
    count = 0
    with open(txtfile, 'a') as f:
        timestamp = datetime.now().strftime("%d-%m-%Y_%Hh%M")
        f.write(timestamp + "\n")
        for fn in files:
            for looping_mode in modes:
                for looping_lambda in lambdas:
                    print("Percent done: ", 100*count/loops, "%")
                    count += 1
                    bn = os.path.basename(fn).split(".")[0]
                    # load diffuser data
                    lensless_fp = os.path.join(diffuser_dir, fn)
                    diffuser = np.load(lensless_fp)
                    diffuser_prep = diffuser - background
                    diffuser_prep = np.clip(diffuser_prep, a_min=0, a_max=1)
                    diffuser_prep /= np.linalg.norm(diffuser_prep.ravel())
                    start_time = time.time()
                    solver = Recon(diffuser_prep, psf_float, mode=looping_mode, lambda1=looping_lambda)
                    allout = solver.iterate()
                    est = solver.get_estimate()
                    proc_time = time.time() - start_time
                    total_time += proc_time           
                    f.write("\n" + bn + ", Processing time: " + str(proc_time))

        f.write("\n" + "Total processing time Convolve2D: " + str(total_time) + "\n")


if __name__ == "__main__":
    mirflickr_dataset()