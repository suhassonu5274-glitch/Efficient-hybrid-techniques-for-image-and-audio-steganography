"""
cli.py - command-line interface for the hybrid steganography project.

Examples
--------
python cli.py demo
python cli.py evaluate
python cli.py embed-image   cover.png out.png "secret text" --key pass --mode dct
python cli.py extract-image out.png --key pass --mode dct
python cli.py embed-audio   cover.wav out.wav --message-file secret.txt --key pass --mode lsb
python cli.py extract-audio out.wav --key pass --mode lsb --out recovered.txt
"""
import argparse
import os
import sys

import audio_stego as aud
import image_stego as img
from stego_core import make_demo_audio, make_demo_image, psnr, snr_db


def get_message(a) -> bytes:
    if a.message_file:
        with open(a.message_file, "rb") as f:
            return f.read()
    if a.message is None:
        sys.exit("Give a message or --message-file.")
    return a.message.encode()


def show_result(data: bytes, out_path):
    if out_path:
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Recovered {len(data)} bytes -> {out_path}")
    else:
        print(data.decode(errors="replace"))


def run_demo():
    os.makedirs("demo_out", exist_ok=True)
    cov_png, cov_wav = "demo_out/cover.png", "demo_out/cover.wav"
    make_demo_image(cov_png)
    make_demo_audio(cov_wav)
    msg = b"Hybrid steganography demo: compressed + encrypted + adaptive embedding."
    key = "demo-key"
    for mode in ("lsb", "dct"):
        c, s = img.embed(cov_png, f"demo_out/stego_{mode}.png", msg, key, mode)
        ok = img.extract(f"demo_out/stego_{mode}.png", key, mode) == msg
        print(f"[IMAGE-{mode.upper()}] recovered={ok}  PSNR={psnr(c, s, 255):.2f} dB")
        c, s = aud.embed(cov_wav, f"demo_out/stego_{mode}.wav", msg, key, mode)
        ok = aud.extract(f"demo_out/stego_{mode}.wav", key, mode) == msg
        print(f"[AUDIO-{mode.upper()}] recovered={ok}  SNR={snr_db(c, s):.2f} dB")
    try:
        img.extract("demo_out/stego_lsb.png", "wrong-key", "lsb")
    except ValueError as e:
        print(f"[WRONG KEY] rejected: {e}")


def main():
    p = argparse.ArgumentParser(description="Hybrid image & audio steganography")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run a self-contained demo")
    sub.add_parser("evaluate", help="run experiments and write graphs to ./results")

    for name, kind in (("embed-image", "image"), ("embed-audio", "audio")):
        s = sub.add_parser(name)
        s.add_argument("cover")
        s.add_argument("output")
        s.add_argument("message", nargs="?")
        s.add_argument("--message-file")
        s.add_argument("--key", required=True)
        s.add_argument("--mode", choices=["lsb", "dct"], default="lsb")
        s.add_argument("--delta", type=float, default=None)
        if kind == "image":
            s.add_argument("--threshold", type=int, default=img.LSB_THRESHOLD)
        else:
            s.add_argument("--amp", type=int, default=aud.LSB_AMP)
    for name, kind in (("extract-image", "image"), ("extract-audio", "audio")):
        s = sub.add_parser(name)
        s.add_argument("stego")
        s.add_argument("--key", required=True)
        s.add_argument("--mode", choices=["lsb", "dct"], default="lsb")
        s.add_argument("--delta", type=float, default=None)
        s.add_argument("--out")
        if kind == "image":
            s.add_argument("--threshold", type=int, default=img.LSB_THRESHOLD)
        else:
            s.add_argument("--amp", type=int, default=aud.LSB_AMP)

    a = p.parse_args()
    try:
        if a.cmd == "demo":
            run_demo()
        elif a.cmd == "evaluate":
            import evaluate
            evaluate.run()
        elif a.cmd == "embed-image":
            c, s = img.embed(a.cover, a.output, get_message(a), a.key, a.mode,
                             a.threshold, a.delta or img.DCT_DELTA)
            print(f"Saved {a.output} | PSNR = {psnr(c, s, 255):.2f} dB")
        elif a.cmd == "embed-audio":
            c, s = aud.embed(a.cover, a.output, get_message(a), a.key, a.mode,
                             a.amp, a.delta or aud.DCT_DELTA)
            print(f"Saved {a.output} | SNR = {snr_db(c, s):.2f} dB")
        elif a.cmd == "extract-image":
            show_result(img.extract(a.stego, a.key, a.mode, a.threshold,
                                    a.delta or img.DCT_DELTA), a.out)
        elif a.cmd == "extract-audio":
            show_result(aud.extract(a.stego, a.key, a.mode, a.amp,
                                    a.delta or aud.DCT_DELTA), a.out)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
