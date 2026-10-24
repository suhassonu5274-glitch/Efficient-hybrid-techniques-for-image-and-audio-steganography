"""
stego_core.py - shared building blocks for the hybrid steganography project.

Pipeline used by BOTH the image and the audio module:
    message --zlib--> compressed --Fernet(AES-128-CBC + HMAC)--> encrypted
            --> [32-bit length header + bits] --> scattered into the cover
              at key-seeded pseudo-random positions.
"""
import base64
import hashlib
import math
import struct
import wave
import zlib

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

HEADER_BITS = 32
MAX_PAYLOAD_BYTES = 50_000_000
NO_DATA_MSG = "No valid hidden data found (wrong key, wrong mode/parameters, or no payload)."


# --------------------------------------------------------------------------
# Payload packing  (compression + encryption + header)
# --------------------------------------------------------------------------
def _fernet(key: str) -> Fernet:
    digest = hashlib.sha256(key.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(b"pos|" + key.encode()).digest()[:8], "big")


def pack_payload(message: bytes, key: str) -> np.ndarray:
    """message -> compress -> encrypt -> 32-bit length header -> bit array."""
    blob = _fernet(key).encrypt(zlib.compress(message, 9))
    data = struct.pack(">I", len(blob)) + blob
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def unpack_payload(read_bits, key: str) -> bytes:
    """read_bits(start, count) must return a uint8 array of `count` bits taken
    from the key-scattered embedding positions, starting at bit `start`."""
    length = struct.unpack(">I", np.packbits(read_bits(0, HEADER_BITS)).tobytes())[0]
    if length <= 0 or length > MAX_PAYLOAD_BYTES:
        raise ValueError(NO_DATA_MSG)
    blob = np.packbits(read_bits(HEADER_BITS, length * 8)).tobytes()
    try:
        return zlib.decompress(_fernet(key).decrypt(blob))
    except (InvalidToken, zlib.error):
        raise ValueError(NO_DATA_MSG)


def scatter_order(slots: np.ndarray, key: str) -> np.ndarray:
    """Key-seeded pseudo-random permutation of the available embedding slots."""
    return np.random.default_rng(_seed(key)).permutation(slots)


# --------------------------------------------------------------------------
# Quantization Index Modulation (QIM) - used by the transform-domain modes
# --------------------------------------------------------------------------
def qim_embed(values: np.ndarray, bits: np.ndarray, delta: float) -> np.ndarray:
    """Move each value to the nearest multiple of `delta` whose index parity == bit."""
    x = values / delta
    q = np.rint(x).astype(np.int64)
    wrong = (q & 1) != bits.astype(np.int64)
    step = np.where(x >= q, 1, -1)
    q = np.where(wrong, q + step, q)
    return q * delta


def qim_extract(values: np.ndarray, delta: float) -> np.ndarray:
    return (np.rint(values / delta).astype(np.int64) & 1).astype(np.uint8)


def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix: coeffs = M @ x ; x = M.T @ coeffs."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0, :] = np.sqrt(1.0 / n)
    return m


# --------------------------------------------------------------------------
# Quality metrics
# --------------------------------------------------------------------------
def mse(a, b) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a, b, peak: float) -> float:
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * math.log10(peak ** 2 / m)


def snr_db(original, stego) -> float:
    o = original.astype(np.float64)
    n = np.sum((o - stego.astype(np.float64)) ** 2)
    return float("inf") if n == 0 else 10 * math.log10(np.sum(o ** 2) / n)


# --------------------------------------------------------------------------
# Synthetic demo covers (so the project runs without any external files)
# --------------------------------------------------------------------------
def make_demo_image(path: str, size: int = 512, seed: int = 1) -> None:
    from PIL import Image
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = 128 + 40 * np.sin(x / 17.0) + 30 * np.cos(y / 23.0)
    chans = [base + 15 * np.sin((x + y) / 11.0 * k) + rng.normal(0, 6, (size, size))
             for k in (1.0, 1.3, 0.7)]
    img = np.clip(np.stack(chans, axis=2), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def make_demo_audio(path: str, seconds: int = 15, sr: int = 44100, seed: int = 2) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(sr * seconds) / sr
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t)
    sig = env * (8000 * np.sin(2 * np.pi * 330 * t)
                 + 4000 * np.sin(2 * np.pi * 660 * t + 1.0)
                 + 2500 * np.sin(2 * np.pi * 1250 * t)) + rng.normal(0, 200, t.size)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.clip(sig, -32768, 32767).astype("<i2").tobytes())
