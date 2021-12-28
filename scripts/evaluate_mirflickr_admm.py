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
import time
import os
import pathlib as plib
import click
from datetime import datetime
from diffcam.io import load_psf
import numpy as np
from diffcam.util import print_image_info
from PIL import Image
from diffcam.mirflickr import ADMM_MIRFLICKR, postprocess
from diffcam.metric import mse, psnr, ssim, lpips


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

    # -- create ADMM object
    recon = ADMM_MIRFLICKR(psf_float)
    print("\nLooping through files...")
    mse_scores = []
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    start_total_time = time.time()
    mode = "admm"
    mu1=1e-6
    mu2=1e-5
    mu3=4e-5

    txtfile = "metricsdata\\admm_metrics_flickrdata.txt"
    with open(txtfile, 'a') as f:
        timestamp = datetime.now().strftime("%d-%m-%Y_%Hh%M")
        f.write(timestamp)
        for fn in files:
            start_time = time.time()
            f.write("\n")
            bn = os.path.basename(fn).split(".")[0]
            print(f"\n{bn}")

            # load diffuser data
            lensless_fp = os.path.join(diffuser_dir, fn)
            diffuser = np.load(lensless_fp)
            diffuser_prep = diffuser - background
            diffuser_prep = np.clip(diffuser_prep, a_min=0, a_max=1)
            diffuser_prep /= np.linalg.norm(diffuser_prep.ravel())
            recon.set_data(diffuser_prep)
            est = recon.apply(n_iter=n_iter, plot=False)

            if save:
                np.save(os.path.join(save, f"{bn}.npy"), est)
                # viewable data
                output_fn = f"plots\\whole_dataset\\admm\\{fn}.tiff"
                est_norm = est / est.max()
                image_data = (est_norm * 255).astype(np.uint8)
                im = Image.fromarray(image_data)
                im.save(output_fn)

            # compute scores
            lensed_fp = os.path.join(lensed_dir, fn)
            lensed = np.load(lensed_fp)
            lensed = postprocess(lensed)
            mse_scores.append(mse(lensed, est))
            psnr_scores.append(psnr(lensed, est))
            ssim_scores.append(ssim(lensed, est))
            lpips_scores.append(lpips(lensed, est))
            proc_time = time.time() - start_time
            with open(txtfile[:-3] + 'csv', 'a') as fi:
                fi.write(', '.join([str(mse_scores[-1]), str(psnr_scores[-1]), str(ssim_scores[-1]),
                                    str(lpips_scores[-1]), str(proc_time), mode, str(mu1), str(mu2), 
                                    str(mu3)])+'\n')
            mse_data =  "MSE: " + str(mse_scores[-1])
            psnr_data = "PSNR: " + str(psnr_scores[-1])
            ssim_data = "SSIM: " + str(ssim_scores[-1])
            lpips_data = "LPIPS: " + str(lpips_scores[-1])
            explanatory_line = "File: " + bn + ", mode: " + mode + ", mu1: " + \
                    str(mu1) + ", mu2: " + str(mu2) + ", mu3: " + str(mu3) + ", process time: " + str(proc_time)
            f.writelines([explanatory_line + "\n", mse_data + "\n", psnr_data + "\n",
                          ssim_data + "\n", lpips_data + "\n"])
        mse_data =  "MSE: " + str(np.mean(mse_scores))
        psnr_data = "PSNR: " + str(np.mean(psnr_scores))
        ssim_data = "SSIM: " + str(np.mean(ssim_scores))
        lpips_data = "LPIPS: " + str(np.mean(lpips_scores))
        total_time = time.time() - start_total_time
        f.writelines(["Total processing time: " + str(total_time) + "\n", "Mean scores:" + "\n", mse_data + "\n", psnr_data + "\n",
                      ssim_data + "\n", lpips_data + "\n"])

    if save:
        print(f"\nReconstructions saved to : {save}")

    print("\nMSE (avg)", np.mean(mse_scores))
    print("PSNR (avg)", np.mean(psnr_scores))
    print("SSIM (avg)", np.mean(ssim_scores))
    print("LPIPS (avg)", np.mean(lpips_scores))


if __name__ == "__main__":
    mirflickr_dataset()
