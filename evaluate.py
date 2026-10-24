"""
evaluate.py - experiments for the project report.

Produces in ./results:
  image_quality.png   PSNR vs payload size (LSB vs DCT)
  audio_quality.png   SNR  vs payload size (LSB vs DCT)
  report.txt          quality numbers, noise-robustness table, chi-square values
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import audio_stego as aud
import image_stego as img
from stego_core import make_demo_audio, make_demo_image, psnr, snr_db

OUT = "results"
KEY = "eval-key"


def chi_square_p(channel: np.ndarray) -> float:
    """Westfeld-Pfitzner chi-square attack on LSB pairs-of-values.
    Returns a p-value; HIGHER means the histogram looks more 'LSB-embedded'."""
    h = np.bincount(channel.ravel(), minlength=256).astype(float)
    a, b = h[0::2], h[1::2]
    e = (a + b) / 2
    m = e > 5
    chi = ((a[m] - e[m]) ** 2 / e[m]).sum()
    k = int(m.sum()) - 1
    z = ((chi / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2))


def attack_image(src, dst, sigma, seed=0):
    a = np.array(Image.open(src).convert("RGB"), dtype=np.float64)
    a += np.random.default_rng(seed).normal(0, sigma, a.shape)
    Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8)).save(dst)


def attack_audio(src, dst, sigma, seed=0):
    params, s = aud.read_wav(src)
    n = s.astype(np.float64) + np.random.default_rng(seed).normal(0, sigma, s.shape)
    aud.write_wav(dst, params, np.clip(np.rint(n), -32768, 32767).astype(np.int16))


def survives(fn) -> str:
    try:
        return "OK  " if fn() else "FAIL"
    except ValueError:
        return "FAIL"


def run():
    os.makedirs(OUT, exist_ok=True)
    cover_png, cover_wav = f"{OUT}/cover.png", f"{OUT}/cover.wav"
    make_demo_image(cover_png)
    make_demo_audio(cover_wav)
    rng = np.random.default_rng(0)
    lines = []

    # ---- 1. Image quality vs payload ---------------------------------------
    sizes = [100, 300, 600, 1000, 1500]
    res = {"lsb": [], "dct": []}
    for n in sizes:
        msg = rng.bytes(n)
        for mode in res:
            c, s = img.embed(cover_png, f"{OUT}/tmp_{mode}.png", msg, KEY, mode)
            res[mode].append(psnr(c, s, 255))
    plt.figure(figsize=(6, 4))
    for mode, lab in (("lsb", "Edge-adaptive LSB"), ("dct", "DCT-QIM")):
        plt.plot(sizes, res[mode], marker="o", label=lab)
    plt.xlabel("Payload size (bytes, incompressible)")
    plt.ylabel("PSNR (dB)")
    plt.title("Image quality vs payload")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/image_quality.png", dpi=150)
    plt.close()
    lines.append("IMAGE PSNR (dB) vs payload bytes")
    lines.append("bytes      " + "  ".join(f"{n:>7}" for n in sizes))
    for mode in res:
        lines.append(f"{mode.upper():<10} " + "  ".join(f"{v:7.2f}" for v in res[mode]))

    # ---- 2. Audio quality vs payload ---------------------------------------
    asizes = [20, 100, 200, 300]
    ares = {"lsb": [], "dct": []}
    for n in asizes:
        msg = rng.bytes(n)
        for mode in ares:
            c, s = aud.embed(cover_wav, f"{OUT}/tmp_{mode}.wav", msg, KEY, mode)
            ares[mode].append(snr_db(c, s))
    plt.figure(figsize=(6, 4))
    for mode, lab in (("lsb", "Amplitude-adaptive LSB"), ("dct", "DCT-QIM")):
        plt.plot(asizes, ares[mode], marker="o", label=lab)
    plt.xlabel("Payload size (bytes, incompressible)")
    plt.ylabel("SNR (dB)")
    plt.title("Audio quality vs payload")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/audio_quality.png", dpi=150)
    plt.close()
    lines += ["", "AUDIO SNR (dB) vs payload bytes",
              "bytes      " + "  ".join(f"{n:>7}" for n in asizes)]
    for mode in ares:
        lines.append(f"{mode.upper():<10} " + "  ".join(f"{v:7.2f}" for v in ares[mode]))

    # ---- 3. Noise robustness -----------------------------------------------
    msg = b"Robustness test message. " * 8
    lines += ["", "ROBUSTNESS to additive Gaussian noise (message recovered?)",
              "Image (sigma in pixel levels):"]
    sig_i = [0, 1, 2, 3]
    lines.append("sigma      " + "  ".join(f"{s:>5}" for s in sig_i))
    for mode in ("lsb", "dct"):
        img.embed(cover_png, f"{OUT}/rob_{mode}.png", msg, KEY, mode)
        row = []
        for sg in sig_i:
            attack_image(f"{OUT}/rob_{mode}.png", f"{OUT}/atk.png", sg)
            row.append(survives(lambda: img.extract(f"{OUT}/atk.png", KEY, mode) == msg))
        lines.append(f"{mode.upper():<10} " + "  ".join(f"{r:>5}" for r in row))
    lines.append("Audio (sigma in sample units, int16):")
    sig_a = [0, 5, 10, 20]
    lines.append("sigma      " + "  ".join(f"{s:>5}" for s in sig_a))
    for mode in ("lsb", "dct"):
        aud.embed(cover_wav, f"{OUT}/rob_{mode}.wav", msg, KEY, mode)
        row = []
        for sg in sig_a:
            attack_audio(f"{OUT}/rob_{mode}.wav", f"{OUT}/atk.wav", sg)
            row.append(survives(lambda: aud.extract(f"{OUT}/atk.wav", KEY, mode) == msg))
        lines.append(f"{mode.upper():<10} " + "  ".join(f"{r:>5}" for r in row))

    # ---- 4. Chi-square steganalysis ----------------------------------------
    lines += ["", "CHI-SQUARE (pairs-of-values) p-value on blue channel; higher = looks embedded"]
    cover_arr = np.array(Image.open(cover_png).convert("RGB"))
    lines.append(f"cover image          : {chi_square_p(cover_arr[:, :, 2]):.4f}")
    c, s = img.embed(cover_png, f"{OUT}/tmp.png", rng.bytes(1500), KEY, "lsb")
    lines.append(f"stego LSB (1500 B)   : {chi_square_p(s[:, :, 2]):.4f}")
    c, s = img.embed(cover_png, f"{OUT}/tmp.png", rng.bytes(1500), KEY, "dct")
    lines.append(f"stego DCT (1500 B)   : {chi_square_p(s[:, :, 2]):.4f}")
    flat = cover_arr.reshape(-1).copy()
    bits = rng.integers(0, 2, flat.size, dtype=np.uint8)
    flat = (flat & 0xFE) | bits
    lines.append(f"naive full-LSB (100%): {chi_square_p(flat.reshape(cover_arr.shape)[:, :, 2]):.4f}"
                 "   <- classic detectable baseline")

    for f in os.listdir(OUT):
        if f.startswith(("tmp", "atk", "rob_")):
            os.remove(os.path.join(OUT, f))
    text = "\n".join(lines)
    with open(f"{OUT}/report.txt", "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nGraphs and report saved in ./{OUT}/")


if __name__ == "__main__":
    run()
