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
import pathlib as plib
import time
from datetime import datetime

import click
import matplotlib.pyplot as plt
import numpy as np
from diffcam.io import load_psf
from diffcam.metric import lpips, mse, psnr, ssim
from diffcam.mirflickr import postprocess
from diffcam.plot import plot_image
from diffcam.recon import Recon
from diffcam.util import print_image_info
from PIL import Image


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
    "--single_psf",
    is_flag=True,
    help="Whether to take PSF as sum of RGB channels.",
)
@click.option(
    "--save",
    is_flag=True,
    help="Whether to save reconstructions.",
)
def mirflickr_dataset(data, n_files, single_psf, save):
    """
    This function is used for hyperparameter tuning, metrics calculation and reconstruction of
    images.
    Parameters
    ----------
    data : str
        File path to raw data measured.
    n_files : int
        Number of files of raw data to reconstruct. Equals the number of performed reconstructions.
    single_psf : bool
        Whether to take PSF as sum of RGB channels.
    save : bool
        Whether to save reconstructions.
    Returns
    -------
    None.
    """
    assert data is not None

    dataset_dir = os.path.join(data, "dataset")
    if os.path.isdir(dataset_dir):
        diffuser_dir = os.path.join(dataset_dir, "diffuser_images")
        lensed_dir = os.path.join(dataset_dir, "ground_truth_lensed")
    else:
        diffuser_dir = os.path.join(data, "diffuser")
        lensed_dir = os.path.join(data, "lensed")
    psf_fp = os.path.join(data, "psf.tiff")
    downsample = 4  # has to be this for collected data!

    # determine files
    files = glob.glob(diffuser_dir + "/*.npy")
    if n_files:
        files = files[:n_files]
    files = [os.path.basename(fn) for fn in files]
    print("Number of files : ", len(files))

    # -- prepare PSF
    print("\nPrepared PSF data")
    psf_float, background = load_psf(
        psf_fp,
        downsample=downsample,
        return_float=True,
        return_bg=True,
        bg_pix=(0, 15),
        single_psf=single_psf,
    )
    print_image_info(psf_float)

    if save:
        timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
        save = "admm_mirflickr" + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)

    mse_scores = []
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    ####### CHANGE TO WHERE YOU WANT YOUR OUTPUT #######
    local_dir = '../all_recon'
    ######### UNCOMMENT THE YOUR LINES OF CODE #########
    ## Iskander's code
    modes = ['dct']
    lambdas = [1e-7,
               1e-6,
               1e-5,
               1e-4,
               1e-3,
               1e-2,
               1e-1,
               5e-1]
    ######### choose the delta to use for Huber mode ########
    huber_delta = [1.5]
    txtfile = 'metrics_flickrdata.txt'
    """  command to run (stand in DiffuserCam when you run)
    python scripts/evaluate_mirflickr_all.py --data subset_mir_flickr_dataset/nn
    """

    ################################################
    start_total_time = time.time()
    with open(txtfile, 'a') as f:
        timestamp = datetime.now().strftime("%d-%m-%Y_%Hh%M")
        f.write(timestamp + "\n")
        for fn in files:
            f.write("\n")
            for looping_mode in modes:
                f.write("\n")
                for looping_lambda in lambdas:
                    f.write("\n")
                    looping_delta = huber_delta[0]
                    start_time = time.time()
                    bn = os.path.basename(fn).split(".")[0]
                    print(f"\n{bn}")
                    # load diffuser data
                    lensless_fp = os.path.join(diffuser_dir, fn)
                    diffuser = np.load(lensless_fp)
                    diffuser_prep = diffuser - background
                    diffuser_prep = np.clip(diffuser_prep, a_min=0, a_max=1)
                    diffuser_prep /= np.linalg.norm(diffuser_prep.ravel())

                    solver = Recon(diffuser_prep, psf_float, mode=looping_mode,
                                   lambda1=looping_lambda, huber_delta = looping_delta)
                    _ = solver.iterate()
                    est = solver.get_estimate()

                    if save:
                        np.save(os.path.join(save, f"{bn}.npy"), est)
                        # viewable data
                        output_fn = os.path.join(save, f"{bn}.tif")
                        est_norm = est / est.max()
                        image_data = (est_norm * 255).astype(np.uint8)
                        im = Image.fromarray(image_data)
                        im.save(output_fn)

                    # compute scores
                    lensed_fp = os.path.join(lensed_dir, fn)
                    lensed = np.load(lensed_fp)
                    lensed = postprocess(lensed)
                    est = postprocess(est)
                    plot_image(est)

                    mse_scores.append(mse(lensed, est))
                    psnr_scores.append(psnr(lensed, est))
                    ssim_scores.append(ssim(lensed, est))
                    lpips_scores.append(lpips(lensed, est))
                    proc_time = time.time() - start_time
                    # write metric data to txt file
                    mse_data =  "MSE: " + str(mse_scores[-1])
                    psnr_data = "PSNR: " + str(psnr_scores[-1])
                    ssim_data = "SSIM: " + str(ssim_scores[-1])
                    lpips_data = "LPIPS: " + str(lpips_scores[-1])

                    with open(txtfile[:-3] + 'csv', 'a') as fi:
                        fi.write(', '.join([str(mse_scores[-1]), str(psnr_scores[-1]),
                                            str(ssim_scores[-1]), str(lpips_scores[-1]),
                                            str(proc_time), looping_mode, str(looping_lambda),
                                            str(looping_delta)])+'\n')

                    # handle extra parameter with Huber
                    if looping_delta == 0:
                        huber_txtstring = ""
                        huber_filestring = ""
                    else:
                        f.write("\n")
                        huber_txtstring = ", Huber delta: " + str(looping_delta)
                        huber_filestring = '_huber_delta_' + str(looping_delta)


                    explanatory_line = "File: " + bn + ", " + "mode: " + looping_mode + ", " + \
                                       "lambda: " + str(looping_lambda) + huber_txtstring + \
                                        ", process time: " + str(proc_time)
                    f.writelines([explanatory_line + "\n", mse_data + "\n", psnr_data + "\n",
                                    ssim_data + "\n", lpips_data + "\n"])
                    # save images in selected folder
                    iteration_variant = bn + '_' + looping_mode + '_' +  str(looping_lambda) + \
                        huber_filestring + '_proc_time_' + str(proc_time)
                    iteration_variant = iteration_variant.replace('.', '_')
                    plt.savefig(local_dir + iteration_variant + '.tiff')
                    plt.close('all')
        print("\nMSE (avg)", np.mean(mse_scores))
        print("PSNR (avg)", np.mean(psnr_scores))
        print("SSIM (avg)", np.mean(ssim_scores))
        print("LPIPS (avg)", np.mean(lpips_scores))
        mse_data =  "MSE: " + str(np.mean(mse_scores))
        psnr_data = "PSNR: " + str(np.mean(psnr_scores))
        ssim_data = "SSIM: " + str(np.mean(ssim_scores))
        lpips_data = "LPIPS: " + str(np.mean(lpips_scores))
        total_time = time.time() - start_total_time
        f.writelines(["Total processing time: " + str(total_time) + "\n", "Mean scores:" + "\n",
                      mse_data + "\n", psnr_data + "\n", ssim_data + "\n", lpips_data + "\n"])

    if save:
        print(f"\nReconstructions saved to : {save}")


if __name__ == "__main__":
    mirflickr_dataset()
