"""
audio_stego.py - hybrid audio steganography for 16-bit PCM WAV files.

  mode="lsb"  Amplitude-adaptive LSB. Bits go only into samples that are loud
              enough (silence is skipped because noise there is audible).
              High capacity and very high SNR, but fragile.

  mode="dct"  Frequency-domain QIM. Channel 0 is cut into 1024-sample frames,
              each frame is DCT-transformed, and bits are hidden by quantizing
              8 mid-frequency bins. Frames that are near-silent are skipped.
              Lower SNR than LSB, but survives mild additive noise.

Output must stay lossless (WAV). MP3/AAC re-encoding destroys the payload.
"""
import wave

import numpy as np

from stego_core import (NO_DATA_MSG, dct_matrix, pack_payload, qim_embed,
                        qim_extract, scatter_order, unpack_payload)

LSB_AMP = 16
DCT_DELTA = 120.0
MIN_ENERGY = 1e6
FRAME = 1024
BINS = [40, 44, 48, 52, 56, 60, 64, 68]
_LOW = slice(1, 40)          # unmodified bins used to decide frame eligibility
_M = dct_matrix(FRAME)


def read_wav(path):
    with wave.open(path, "rb") as w:
        params = w.getparams()
        if params.sampwidth != 2:
            raise ValueError("Only 16-bit PCM WAV is supported.")
        raw = w.readframes(params.nframes)
    samples = np.frombuffer(raw, dtype="<i2").copy().reshape(-1, params.nchannels)
    return params, samples


def write_wav(path, params, samples):
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(samples.reshape(-1).astype("<i2").tobytes())


# ------------------------------- LSB mode ---------------------------------
def _lsb_slots(flat, amp):
    return np.flatnonzero(np.abs(flat.astype(np.int32) >> 1) >= amp)


def _embed_lsb(samples, bits, key, amp):
    flat = samples.reshape(-1).copy()
    slots = _lsb_slots(flat, amp)
    if len(bits) > len(slots):
        raise ValueError(f"Payload needs {len(bits)} bits but only {len(slots)} samples are "
                         f"loud enough. Lower --amp or use a longer/louder file.")
    pos = scatter_order(slots, key)[: len(bits)]
    flat[pos] = (flat[pos] & ~1) | bits.astype(np.int16)
    return flat.reshape(samples.shape)


def _extract_lsb(samples, key, amp):
    flat = samples.reshape(-1)
    order = scatter_order(_lsb_slots(flat, amp), key)

    def read_bits(start, count):
        if start + count > len(order):
            raise ValueError(NO_DATA_MSG)
        return (flat[order[start:start + count]] & 1).astype(np.uint8)

    return unpack_payload(read_bits, key)


# ------------------------------- DCT mode ---------------------------------
def _dct_prepare(samples, min_energy):
    x = samples[:, 0]
    nf = len(x) // FRAME
    frames = x[: nf * FRAME].astype(np.float64).reshape(nf, FRAME)
    d = frames @ _M.T
    eligible = (d[:, _LOW] ** 2).sum(axis=1) >= min_energy
    slots = np.flatnonzero(np.repeat(eligible, len(BINS)))
    return d, nf, slots


def _embed_dct(samples, bits, key, delta, min_energy):
    d, nf, slots = _dct_prepare(samples, min_energy)
    if len(bits) > len(slots):
        raise ValueError(f"Payload needs {len(bits)} bits but DCT capacity is only "
                         f"{len(slots)} bits. Use a longer file or a shorter message.")
    pos = scatter_order(slots, key)[: len(bits)]
    vals = d[:, BINS]
    flat = vals.reshape(-1)
    flat[pos] = qim_embed(flat[pos], bits, delta)
    d[:, BINS] = flat.reshape(vals.shape)
    rec = np.clip(np.rint((d @ _M).reshape(-1)), -32768, 32767).astype(np.int16)
    out = samples.copy()
    out[: nf * FRAME, 0] = rec
    return out


def _extract_dct(samples, key, delta, min_energy):
    d, _, slots = _dct_prepare(samples, min_energy)
    order = scatter_order(slots, key)
    vals = d[:, BINS].reshape(-1)

    def read_bits(start, count):
        if start + count > len(order):
            raise ValueError(NO_DATA_MSG)
        return qim_extract(vals[order[start:start + count]], delta)

    return unpack_payload(read_bits, key)


# ------------------------------ public API --------------------------------
def extract_array(samples, key, mode="lsb", amp=LSB_AMP, delta=DCT_DELTA,
                  min_energy=MIN_ENERGY) -> bytes:
    if mode == "lsb":
        return _extract_lsb(samples, key, amp)
    if mode == "dct":
        return _extract_dct(samples, key, delta, min_energy)
    raise ValueError("mode must be 'lsb' or 'dct'")


def embed(cover_path, out_path, message: bytes, key: str, mode="lsb", amp=LSB_AMP,
          delta=DCT_DELTA, min_energy=MIN_ENERGY):
    """Hide `message` in a WAV file. Returns (cover_samples, stego_samples) flattened."""
    params, cover = read_wav(cover_path)
    bits = pack_payload(message, key)
    if mode == "lsb":
        stego = _embed_lsb(cover, bits, key, amp)
    elif mode == "dct":
        stego = _embed_dct(cover, bits, key, delta, min_energy)
    else:
        raise ValueError("mode must be 'lsb' or 'dct'")
    if extract_array(stego, key, mode, amp, delta, min_energy) != message:
        raise ValueError("Self-check failed. Try a larger --delta or a different cover.")
    write_wav(out_path, params, stego)
    return cover.reshape(-1), stego.reshape(-1)


def extract(stego_path, key: str, mode="lsb", amp=LSB_AMP, delta=DCT_DELTA,
            min_energy=MIN_ENERGY) -> bytes:
    _, samples = read_wav(stego_path)
    return extract_array(samples, key, mode, amp, delta, min_energy)
