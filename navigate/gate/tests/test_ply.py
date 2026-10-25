"""rtabmap-export が書き出す PLY の点を読み、PointCloud2 の並びへ詰める（純粋関数）の仕様。

rtabmap-export の PLY は binary_little_endian で、vertex の後ろに face（0 個）と camera の要素が続く。
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from navgate.ply import interleave_xyzrgb, pack_rgb, read_ply_vertices

VERTEX_PROPERTIES = [
    ("float", "x"),
    ("float", "y"),
    ("float", "z"),
    ("float", "nx"),
    ("float", "ny"),
    ("float", "nz"),
    ("uchar", "red"),
    ("uchar", "green"),
    ("uchar", "blue"),
    ("float", "curvature"),
]
PACK = {"float": "f", "uchar": "B"}


def _rtabmap_ply(rows: list[tuple], fmt: str = "binary_little_endian") -> bytes:
    """rtabmap-export と同じ並びの PLY を作る。camera の要素は中身を読まないので 0 で埋める。"""
    header = [
        "ply",
        f"format {fmt} 1.0",
        "comment PCL generated",
        f"element vertex {len(rows)}",
        *(f"property {t} {n}" for t, n in VERTEX_PROPERTIES),
        "element face 0",
        "property list uchar int vertex_indices",
        "element camera 1",
        "property float view_px",
        "property float view_py",
        "end_header",
    ]
    body = b"".join(struct.pack("<" + "".join(PACK[t] for t, _ in VERTEX_PROPERTIES), *r) for r in rows)
    return ("\n".join(header) + "\n").encode() + body + struct.pack("<ff", 0.0, 0.0)


ROWS = [
    (1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 255, 0, 10, 0.5),
    (-0.5, 0.25, 0.0, 1.0, 0.0, 0.0, 0, 128, 255, 0.0),
]


def test_reads_every_vertex_property_by_name() -> None:
    v = read_ply_vertices(_rtabmap_ply(ROWS))
    np.testing.assert_allclose(v["x"], [1.0, -0.5])
    np.testing.assert_allclose(v["z"], [3.0, 0.0])
    assert v["red"].tolist() == [255, 0]
    assert v["blue"].tolist() == [10, 255]
    np.testing.assert_allclose(v["curvature"], [0.5, 0.0])


def test_elements_after_the_vertices_are_ignored() -> None:
    v = read_ply_vertices(_rtabmap_ply(ROWS))
    assert len(v["x"]) == len(ROWS)


def test_rejects_ascii_ply() -> None:
    with pytest.raises(ValueError, match="binary_little_endian"):
        read_ply_vertices(_rtabmap_ply(ROWS, fmt="ascii"))


def test_rejects_a_file_without_vertices_first() -> None:
    data = _rtabmap_ply(ROWS).replace(b"element vertex 2", b"element face 0\nelement vertex 2", 1)
    with pytest.raises(ValueError, match="vertex"):
        read_ply_vertices(data)


def test_pack_rgb_follows_the_ros_convention() -> None:
    # PointCloud2 の rgb は、float32 の 4 バイトに 0x00RRGGBB を入れる
    packed = pack_rgb(
        np.array([255, 0], np.uint8), np.array([0, 128], np.uint8), np.array([10, 255], np.uint8)
    )
    assert packed.dtype == np.float32
    assert packed.view(np.uint32).tolist() == [0xFF000A, 0x0080FF]


def test_interleave_puts_x_y_z_rgb_at_offsets_0_4_8_12() -> None:
    v = read_ply_vertices(_rtabmap_ply(ROWS))
    data = interleave_xyzrgb(v["x"], v["y"], v["z"], pack_rgb(v["red"], v["green"], v["blue"]))
    assert len(data) == 16 * len(ROWS)
    x, y, z, rgb = struct.unpack_from("<fffI", data, 16)
    assert (x, y, z, rgb) == (-0.5, 0.25, 0.0, 0x0080FF)
