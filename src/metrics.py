import numpy as np
from skimage.metrics import structural_similarity
from scipy.stats import chi2 as chi2_dist


def calculate_metrics(cover: np.ndarray, stego: np.ndarray) -> dict:
    c = cover[..., :3].astype(np.float64)
    s = stego[..., :3].astype(np.float64)

    mse = np.mean((c - s) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
    ssim = structural_similarity(cover[..., :3], stego[..., :3], channel_axis=2, data_range=255)

    changed = int(np.sum(cover[..., :3] != stego[..., :3]))
    total_rgb_bytes = cover[..., :3].size

    return {
        'psnr': psnr,
        'ssim': ssim,
        'changed': changed,
        'ratio': changed / total_rgb_bytes
    }


def lsb_distribution(img: np.ndarray) -> float:
    bits = (img[..., :3].flatten()) & 1
    return float(bits.mean())


def chi2_lsb(img: np.ndarray) -> tuple:
    flat_rgb = img[..., :3].flatten()
    h = np.bincount(flat_rgb, minlength=256).astype(np.float64)
    even, odd = h[0::2], h[1::2]
    n = even + odd
    mask = n > 0
    expected = n[mask] / 2.0
    chi2_val = (np.sum((even[mask] - expected) ** 2 / expected) +
                np.sum((odd[mask] - expected) ** 2 / expected))
    df = max(1, int(mask.sum()) - 1)
    p = float(1.0 - chi2_dist.cdf(chi2_val, df)) if df > 0 else 1.0
    return float(chi2_val), p