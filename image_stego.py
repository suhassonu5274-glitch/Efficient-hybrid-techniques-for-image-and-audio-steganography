"""
image_stego.py - hybrid image steganography.

Two modes (both use compression + AES encryption + key-scattered positions):

  mode="lsb"  Edge-adaptive LSB. Bits go only into pixels in textured/edge
              regions (where changes are hard to see). The edge map is computed
              from the upper 7 bit-planes, so the receiver rebuilds it exactly.
              High capacity, highest PSNR, but fragile (noise destroys it).

  mode="dct"  DCT-domain QIM. The blue channel is split into 8x8 blocks, and bits
              are hidden in mid-frequency DCT coefficients by quantization
              (QIM). Lower PSNR than LSB but survives mild noise.

Stego images MUST be saved losslessly (PNG).
"""
import numpy as np
from PIL import Image

from stego_core import (NO_DATA_MSG, dct_matrix, pack_payload, qim_embed,
                        qim_extract, scatter_order, unpack_payload)

LSB_THRESHOLD = 12
DCT_DELTA = 20.0
DCT_COEFS = [(2, 3), (3, 2), (3, 3), (1, 4), (4, 1), (2, 4)]   # mid-frequency
_RR = [c[0] for c in DCT_COEFS]
_CC = [c[1] for c in DCT_COEFS]
DC_MIN, DC_MAX = 30 * 8, 225 * 8      # skip very dark/bright blocks (clipping risk)
_M = dct_matrix(8)


def _load(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


# ------------------------------- LSB mode ---------------------------------
def _edge_slots(arr: np.ndarray, threshold: int) -> np.ndarray:
    base = (arr & 0xFE).astype(np.int32)
    gray = base.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    mask = np.repeat(((gx + gy) >= threshold)[:, :, None], 3, axis=2)
    return np.flatnonzero(mask.ravel())


def _embed_lsb(arr, bits, key, threshold):
    slots = _edge_slots(arr, threshold)
    if len(bits) > len(slots):
        raise ValueError(f"Payload needs {len(bits)} bits but only {len(slots)} edge "
                         f"positions are available. Lower the threshold or use a bigger/"
                         f"more textured image.")
    pos = scatter_order(slots, key)[: len(bits)]
    flat = arr.ravel().copy()
    flat[pos] = (flat[pos] & 0xFE) | bits
    return flat.reshape(arr.shape)


def _extract_lsb(arr, key, threshold):
    order = scatter_order(_edge_slots(arr, threshold), key)
    flat = arr.ravel()

    def read_bits(start, count):
        if start + count > len(order):
            raise ValueError(NO_DATA_MSG)
        return (flat[order[start:start + count]] & 1).astype(np.uint8)

    return unpack_payload(read_bits, key)


# ------------------------------- DCT mode ---------------------------------
def _to_blocks(ch):
    h, w = ch.shape
    hh, ww = h // 8 * 8, w // 8 * 8
    b = ch[:hh, :ww].astype(np.float64).reshape(hh // 8, 8, ww // 8, 8)
    return b.swapaxes(1, 2).reshape(-1, 8, 8), hh, ww


def _from_blocks(blocks, hh, ww):
    return blocks.reshape(hh // 8, ww // 8, 8, 8).swapaxes(1, 2).reshape(hh, ww)


def _dct_prepare(arr):
    blocks, hh, ww = _to_blocks(arr[:, :, 2])
    d = _M @ blocks @ _M.T
    eligible = (d[:, 0, 0] >= DC_MIN) & (d[:, 0, 0] <= DC_MAX)
    slots = np.flatnonzero(np.repeat(eligible, len(DCT_COEFS)))
    return d, hh, ww, slots


def _embed_dct(arr, bits, key, delta):
    d, hh, ww, slots = _dct_prepare(arr)
    if len(bits) > len(slots):
        raise ValueError(f"Payload needs {len(bits)} bits but DCT capacity is only "
                         f"{len(slots)} bits. Use a larger image or a shorter message.")
    pos = scatter_order(slots, key)[: len(bits)]
    vals = d[:, _RR, _CC]
    flat = vals.reshape(-1)
    flat[pos] = qim_embed(flat[pos], bits, delta)
    d[:, _RR, _CC] = flat.reshape(vals.shape)
    rec = _M.T @ d @ _M
    out = arr.copy()
    out[:hh, :ww, 2] = np.clip(np.rint(_from_blocks(rec, hh, ww)), 0, 255).astype(np.uint8)
    return out


def _extract_dct(arr, key, delta):
    d, _, _, slots = _dct_prepare(arr)
    order = scatter_order(slots, key)
    vals = d[:, _RR, _CC].reshape(-1)

    def read_bits(start, count):
        if start + count > len(order):
            raise ValueError(NO_DATA_MSG)
        return qim_extract(vals[order[start:start + count]], delta)

    return unpack_payload(read_bits, key)


# ------------------------------ public API --------------------------------
def extract_array(arr, key, mode="lsb", threshold=LSB_THRESHOLD, delta=DCT_DELTA) -> bytes:
    if mode == "lsb":
        return _extract_lsb(arr, key, threshold)
    if mode == "dct":
        return _extract_dct(arr, key, delta)
    raise ValueError("mode must be 'lsb' or 'dct'")


def embed(cover_path, out_path, message: bytes, key: str, mode="lsb",
          threshold=LSB_THRESHOLD, delta=DCT_DELTA):
    """Hide `message` in the cover image. Returns (cover_array, stego_array)."""
    if not str(out_path).lower().endswith(".png"):
        raise ValueError("Output must be a .png file (lossless).")
    cover = _load(cover_path)
    bits = pack_payload(message, key)
    if mode == "lsb":
        stego = _embed_lsb(cover, bits, key, threshold)
    elif mode == "dct":
        stego = _embed_dct(cover, bits, key, delta)
    else:
        raise ValueError("mode must be 'lsb' or 'dct'")
    # self-check: guarantees the receiver can recover the message
    if extract_array(stego, key, mode, threshold, delta) != message:
        raise ValueError("Self-check failed (likely clipping in very bright/dark areas). "
                         "Try a different cover image or a larger --delta.")
    Image.fromarray(stego).save(out_path, format="PNG")
    return cover, stego


def extract(stego_path, key: str, mode="lsb", threshold=LSB_THRESHOLD, delta=DCT_DELTA) -> bytes:
    return extract_array(_load(stego_path), key, mode, threshold, delta)
