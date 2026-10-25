"""rtabmap-export が書き出す PLY の点を読み、sensor_msgs/PointCloud2 の並び（x、y、z、rgb）へ詰める。"""

from __future__ import annotations

import numpy as np

_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def read_ply_vertices(data: bytes) -> dict[str, np.ndarray]:
    """binary_little_endian の PLY から、vertex の各プロパティを名前ごとの配列で返す。

    vertex が最初の要素であることを求める。後ろの要素（rtabmap-export なら face と camera）は読まない。
    """
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise ValueError("PLY の header が読めない")
    body_start = data.index(b"\n", end) + 1
    fmt = first_element = None
    count = 0
    fields: list[tuple[str, str]] = []
    in_vertex = False
    for line in data[:end].decode("ascii").splitlines()[1:]:
        words = line.split()
        if not words or words[0] == "comment":
            continue
        if words[0] == "format":
            fmt = words[1]
        elif words[0] == "element":
            first_element = first_element or words[1]
            in_vertex = words[1] == "vertex"
            if in_vertex:
                count = int(words[2])
        elif words[0] == "property" and in_vertex:
            if words[1] == "list" or words[1] not in _TYPES:
                raise ValueError(f"vertex のプロパティの型を読めない: {line}")
            fields.append((words[2], _TYPES[words[1]]))
    if fmt != "binary_little_endian":
        raise ValueError(f"binary_little_endian の PLY だけを読む（この PLY は {fmt}）")
    if first_element != "vertex":
        raise ValueError("vertex が最初の要素ではない PLY は読めない")
    vertices = np.frombuffer(data, dtype=np.dtype(fields), count=count, offset=body_start)
    return {name: vertices[name].copy() for name, _ in fields}


def pack_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """PointCloud2 の rgb の規約どおり、float32 の 4 バイトに 0x00RRGGBB を入れる。"""
    packed = (red.astype(np.uint32) << 16) | (green.astype(np.uint32) << 8) | blue.astype(np.uint32)
    return packed.view(np.float32)


def interleave_xyzrgb(x: np.ndarray, y: np.ndarray, z: np.ndarray, rgb: np.ndarray) -> bytes:
    """1 点 16 バイトで、x、y、z、rgb を 0、4、8、12 バイト目に置いた並びを返す。"""
    out = np.empty(len(x), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    out["x"], out["y"], out["z"], out["rgb"] = x, y, z, rgb
    return out.tobytes()
