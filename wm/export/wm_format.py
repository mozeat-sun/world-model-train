""".wm binary weight format — writer (Python side).

Loads magic bytes from spec/spec/constants.json; falls back to hardcoded values.
"""

import json
import struct
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Load magic bytes from spec (if available)
_SPEC_PATH = Path(__file__).parent.parent.parent / "spec" / "spec" / "constants.json"
try:
    with open(_SPEC_PATH) as f:
        _SPEC = json.load(f)
    _MAGIC_DICT = _SPEC["magic_bytes"]
    MAGIC = {k: v.encode('ascii') for k, v in _MAGIC_DICT.items()}
except (FileNotFoundError, KeyError):
    MAGIC = {
        'encoder':   b'WMEN',
        'dynamics':  b'WMDY',
        'planner':   b'WMPL',
        'critic':    b'WMCR',
        'reward':    b'WMRW',
    }


def _hash_name(name: str) -> int:
    """FNV-1a 32-bit hash of tensor name."""
    h = 0x811c9dc5
    for b in name.encode('utf-8'):
        h = (h ^ b) * 0x01000193
        h = h & 0xffffffff
    return h


def write_wm_header(buf: bytearray, magic: bytes, config: dict):
    """Write 64-byte header."""
    buf.extend(magic.ljust(4, b'\x00')[:4])
    buf.extend(struct.pack('<I', 1))                 # version
    buf.extend(struct.pack('<I', config.get('latent_dim', 128)))
    buf.extend(struct.pack('<I', config.get('n_channels', 11)))
    buf.extend(struct.pack('<I', config.get('max_symbols', 100)))
    buf.extend(struct.pack('<I', config.get('window_len', 60)))
    buf.extend(struct.pack('<I', config.get('n_mask_categories', 3)))
    buf.extend(struct.pack('<I', config.get('attn_heads', 4)))
    buf.extend(struct.pack('<I', config.get('gru_layers', 2)))
    buf.extend(struct.pack('<I', config.get('gru_hidden', 256)))
    buf.extend(struct.pack('<I', config.get('predict_horizon', 20)))
    # Reserved + checksum (filled later)
    buf.extend(b'\x00' * (64 - 44))


def write_wm_weights(
    path: str,
    magic: str,
    config: dict,
    tensors: List[Tuple[str, np.ndarray]],
    norm_mean: np.ndarray = None,
    norm_std: np.ndarray = None,
):
    """
    Write a .wm weight file.

    Args:
        path:        output file path
        magic:       one of 'encoder','dynamics','planner','critic','reward'
        config:      dict with model config (latent_dim, etc.)
        tensors:     list of (name, float32_array) in canonical order
        norm_mean:   normalization mean (encoder only)
        norm_std:    normalization std (encoder only)
    """
    magic_bytes = MAGIC.get(magic, b'WMXX')
    buf = bytearray()

    # Header (64 bytes placeholder, actual checksum filled later)
    write_wm_header(buf, magic_bytes, config)

    # Normalization params (encoder only)
    if norm_mean is not None:
        buf.extend(norm_mean.astype(np.float32).tobytes())
    if norm_std is not None:
        buf.extend(norm_std.astype(np.float32).tobytes())

    # Weights section
    weights_start = len(buf)
    for name, arr in tensors:
        data = arr.astype(np.float32)
        name_hash = _hash_name(name)
        buf.extend(struct.pack('<I', name_hash))
        buf.extend(struct.pack('<I', data.size))
        buf.extend(data.tobytes())

    weights_end = len(buf)
    weights_data = bytes(buf[weights_start:weights_end])

    # Compute simple checksum (CRC32-like, for MVP)
    checksum = 0
    for b in weights_data:
        checksum = (checksum * 31 + b) & 0xffffffff

    # Write checksum into header at offset 56
    struct.pack_into('<I', buf, 56, checksum)

    # Footer: SHA256 of weights + total size
    sha = hashlib.sha256(weights_data).digest()
    buf.extend(sha)
    buf.extend(struct.pack('<Q', len(buf) + 32))

    with open(path, 'wb') as f:
        f.write(buf)

    print(f"Wrote {path}: {len(buf)} bytes, {len(tensors)} tensors")
