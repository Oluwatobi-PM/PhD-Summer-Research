"""Small MATLAB v5 MAT-file reader for numeric arrays.

SciPy is used when installed. The fallback parser covers the simple numeric
`baseinfo.mat` and `oilgb.mat` files used by this project.
"""

from __future__ import annotations

from pathlib import Path
import struct
import zlib

import numpy as np


def loadmat_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        from scipy.io import loadmat  # type: ignore

        return {k: v for k, v in loadmat(path, squeeze_me=True).items() if not k.startswith("__")}
    except ImportError:
        pass

    raw = path.read_bytes()
    if len(raw) < 128:
        return {}
    endian = "<" if raw[126:128] == b"IM" else ">"
    arrays: dict[str, np.ndarray] = {}
    offset = 128
    while offset + 8 <= len(raw):
        dtype, nbytes, offset = read_tag(raw, offset, endian)
        payload = raw[offset : offset + nbytes]
        offset += pad8(nbytes)
        if dtype == 15:
            parse_stream(zlib.decompress(payload), endian, arrays)
        elif dtype == 14:
            item = parse_matrix(payload, endian)
            if item is not None:
                name, value = item
                arrays[name] = value
    return arrays


def parse_stream(raw: bytes, endian: str, arrays: dict[str, np.ndarray]) -> None:
    offset = 0
    while offset + 8 <= len(raw):
        dtype, nbytes, offset = read_tag(raw, offset, endian)
        payload = raw[offset : offset + nbytes]
        offset += pad8(nbytes)
        if dtype == 14:
            item = parse_matrix(payload, endian)
            if item is not None:
                name, value = item
                arrays[name] = value


def parse_matrix(payload: bytes, endian: str) -> tuple[str, np.ndarray] | None:
    offset = 0
    _, nbytes, offset = read_tag(payload, offset, endian)
    offset += pad8(nbytes)
    _, nbytes, offset = read_tag(payload, offset, endian)
    dims_raw = payload[offset : offset + nbytes]
    offset += pad8(nbytes)
    dims = struct.unpack(endian + "i" * (nbytes // 4), dims_raw)
    _, nbytes, offset = read_tag(payload, offset, endian)
    name = payload[offset : offset + nbytes].decode("latin1")
    offset += pad8(nbytes)
    dtype, nbytes, offset = read_tag(payload, offset, endian)
    data = payload[offset : offset + nbytes]
    if dtype == 9:
        values = np.frombuffer(data, dtype=endian + "f8")
    elif dtype == 7:
        values = np.frombuffer(data, dtype=endian + "f4")
    elif dtype == 5:
        values = np.frombuffer(data, dtype=endian + "i4")
    elif dtype == 4:
        values = np.frombuffer(data, dtype=endian + "u2")
    elif dtype == 3:
        values = np.frombuffer(data, dtype=endian + "i2")
    elif dtype == 2:
        values = np.frombuffer(data, dtype="u1")
    elif dtype == 1:
        values = np.frombuffer(data, dtype="i1")
    else:
        return None
    shape = tuple(int(d) for d in dims)
    return name, values.reshape(shape, order="F").squeeze().copy()


def read_tag(raw: bytes, offset: int, endian: str) -> tuple[int, int, int]:
    word1, word2 = struct.unpack_from(endian + "II", raw, offset)
    offset += 8
    dtype = word1 & 0xFFFF
    small_nbytes = (word1 >> 16) & 0xFFFF
    if small_nbytes:
        nbytes = small_nbytes
        offset -= 4
    else:
        dtype = word1
        nbytes = word2
    return dtype, nbytes, offset


def pad8(nbytes: int) -> int:
    return nbytes + ((8 - nbytes % 8) % 8)
