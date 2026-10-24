# Efficient Hybrid Techniques for Image and Audio Steganography

A complete, runnable project that hides secret messages inside **images (PNG)** and
**audio (16-bit WAV)** by combining several techniques in one pipeline.

## Setup

```bash
pip install -r requirements.txt        # numpy, pillow, cryptography, matplotlib
python cli.py demo                     # self-contained demo (creates demo_out/)
python cli.py evaluate                 # experiments + graphs (creates results/)
python gui.py                          # desktop GUI (needs tkinter)
```

## Files

| File | Purpose |
|---|---|
| `stego_core.py` | Compression, encryption, header, key-scattered positions, QIM, DCT matrix, PSNR/SNR, demo covers |
| `image_stego.py` | Image modes: `lsb` (edge-adaptive) and `dct` (DCT-domain QIM) |
| `audio_stego.py` | Audio modes: `lsb` (amplitude-adaptive) and `dct` (frequency-domain QIM) |
| `cli.py` | Command-line interface |
| `gui.py` | Tkinter GUI with Image and Audio tabs |
| `evaluate.py` | Quality-vs-payload graphs, noise-robustness table, chi-square check |

## Why it is "hybrid"

1. **Compression (zlib)** - fewer bits to hide, so capacity is used efficiently.
2. **Encryption (Fernet: AES-128-CBC + HMAC-SHA256)** - the payload is unreadable and
   tamper-evident; a wrong key is rejected instead of returning garbage.
3. **Adaptive embedding** - image LSB uses only edge/texture pixels; audio LSB skips
   quiet samples; DCT modes skip very dark/bright blocks and near-silent frames.
4. **Key-seeded pseudo-random scatter** - bit positions are a secret permutation, not sequential.
5. **Two domains** - spatial/time-domain LSB (high capacity, best quality) and
   transform-domain QIM (more robust to noise).

## Usage

```bash
# image
python cli.py embed-image   cover.png stego.png "my secret" --key pass --mode lsb
python cli.py extract-image stego.png --key pass --mode lsb
python cli.py embed-image   cover.png stego.png --message-file secret.txt --key pass --mode dct

# audio
python cli.py embed-audio   cover.wav stego.wav "my secret" --key pass --mode dct
python cli.py extract-audio stego.wav --key pass --mode dct --out recovered.txt
```

The receiver must use the **same key, mode and parameters** (`--threshold`, `--amp`, `--delta`).

## How it works (short version for your viva/report)

**Payload:** `message -> zlib -> Fernet -> [32-bit length | ciphertext bits]`.

**Image LSB:** the edge map `|dx|+|dy| >= threshold` is computed from the top 7 bit-planes,
which LSB changes never touch, so sender and receiver derive the identical set of
eligible pixels. Eligible positions are permuted with a key-seeded RNG; bit *i* of the
payload replaces the LSB at the *i*-th position.

**Image DCT:** the blue channel is split into 8x8 blocks, DCT-transformed, and each
bit is stored in one of 6 mid-frequency coefficients by Quantization Index Modulation:
the coefficient is moved to the nearest multiple of `delta` whose index parity equals the bit.
Extraction rounds `coef/delta` and reads the parity. Noise smaller than `delta/2` is tolerated.

**Audio LSB:** same idea on 16-bit samples with `|sample>>1| >= amp` as the eligibility rule.

**Audio DCT:** 1024-sample frames, orthonormal DCT, QIM on 8 mid-frequency bins
(40..68 step 4). A frame is eligible if the energy of the *unmodified* low bins is above `min_energy`.

Every `embed` call runs a **self-check** (extracts what it just embedded) and fails loudly
rather than producing a file that cannot be decoded.

## Measured results (from `python cli.py evaluate`, synthetic covers)

Image 512x512, payload = incompressible random bytes:

| Payload (bytes) | 100 | 300 | 600 | 1000 | 1500 |
|---|---|---|---|---|---|
| LSB PSNR (dB) | 77.59 | 73.91 | 71.44 | 69.46 | 67.85 |
| DCT PSNR (dB) | 53.14 | 49.67 | 47.09 | 45.09 | 43.47 |

Audio 15 s, 44.1 kHz mono:

| Payload (bytes) | 20 | 100 | 200 | 300 |
|---|---|---|---|---|
| LSB SNR (dB) | 104.45 | 101.60 | 99.45 | 98.15 |
| DCT SNR (dB) | 64.39 | 61.63 | 59.53 | 58.30 |

Recovery after additive Gaussian noise:

| | image sigma 0 / 1 / 2 / 3 | audio sigma 0 / 5 / 10 / 20 |
|---|---|---|
| LSB | OK / FAIL / FAIL / FAIL | OK / FAIL / FAIL / FAIL |
| DCT | OK / OK / OK / FAIL | OK / OK / OK / FAIL |

Trade-off: LSB gives higher quality and capacity; DCT costs 20-40 dB of quality but
survives mild noise.

**Chi-square note:** the evaluation prints Westfeld-Pfitzner p-values, but on the
synthetic demo cover the *unmodified* image already scores 0.94, so that test does not
discriminate here. Use real photographs for any steganalysis claims; do not cite the demo numbers as evidence of undetectability.

## Limitations (state these honestly in your report)

- Stego files must stay lossless: **PNG** and **WAV**. JPEG/MP3 re-compression destroys the data
  (neither mode is designed to survive lossy compression).
- Robustness is limited to mild additive noise; no resistance to cropping, resizing or resampling.
- Capacity depends on the cover: smooth images and quiet audio have fewer eligible positions.
- DCT image mode only uses the blue channel; DCT audio mode only uses channel 0.
- Fernet's encoding adds roughly 60% overhead for tiny messages; compression offsets this for longer text.
- Demo covers are synthetic. For your final results, rerun `evaluate.py` with real photos and speech/music.

## Ideas for extension

- Add error-correcting codes (e.g. Reed-Solomon) to the DCT modes for stronger noise tolerance.
- Use DWT instead of DCT, or embed in all three colour channels.
- Split one payload across an image and an audio file (true cross-media hybrid).
- Run RS analysis / SPA steganalysis on real photos.
