import click
import imageio
import numpy as np
from diffcam.io import load_image
from diffcam.metric import lpips, mse, psnr, ssim
from diffcam.util import resize


@click.command()
@click.option(
    "--recon",
    type=str,
    help="File path to reconstruction.",
)
@click.option(
    "--original",
    type=str,
    help="File path to original file.",
)
def compute_metrics(recon, original):
    """
    The program takes paths to two cropped images as input and computes the metrics. The images 
    used can be screenshots e.g.
    Parameters
    ----------
    recon : string
        File path to reconstruction.
    original : string
        File path to original file.
    Returns
    -------
    None.
    """
    est = np.asarray(imageio.imread(recon))
    est = est/np.amax(est)
    est = np.clip(est, 0, 1)
    est = est[:,:,0:3]
    img = load_image(original)
    img = img / img.max()
    factor = est.shape[1] / img.shape[1]
    img_resize = np.zeros_like(est)
    tmp = resize(img, factor=factor).astype(est.dtype)
    img_resize[: min(est.shape[0], tmp.shape[0]), : min(est.shape[1], tmp.shape[1])] = tmp[
        : min(est.shape[0], tmp.shape[0]), : min(est.shape[1], tmp.shape[1])
    ]
    est = est.astype(np.float32)
    img_resize = img_resize.astype(np.float32)
    print("\nMSE", mse(img_resize, est))
    print("PSNR", psnr(img_resize, est))
    print("SSIM", ssim(img_resize, est))
    print("LPIPS", lpips(img_resize, est))


if __name__ == "__main__":
    compute_metrics()
