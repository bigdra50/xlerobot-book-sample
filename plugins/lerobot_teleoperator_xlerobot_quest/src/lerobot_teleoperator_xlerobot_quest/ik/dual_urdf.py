"""同梱の XLeRobot の URDF を、PyRoki がそのまま読める形に整える。標準ライブラリだけで動く。"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# リポジトリに同梱した XLeRobot の URDF（腕は so101_new_calib、車体と head は公式のメッシュ）。
# plugin は editable で入れる前提なので、リポジトリのパスをそのまま指す
SHIPPED_URDF = Path(__file__).resolve().parents[5] / "urdf" / "xlerobot_so101.urdf"


def prepare_shipped_urdf(urdf_xml: str, *, mesh_dir: str) -> str:
    """同梱 URDF を、PyRoki がそのまま読める形にして返す。

    joint を根元から手先の順に並べ替え、相対パスのメッシュを mesh_dir 基準の絶対パスにする。
    PyRoki は joint の並び順を関節角の並びに使うので、腕ごとに shoulder_pan から wrist_roll の順にそろえる。
    teleop_xr は URDF をキャッシュへ書き出してから読むので、相対パスのままではメッシュが見つからない。
    """
    root = ET.fromstring(urdf_xml)
    _absolutize_mesh_paths(root, mesh_dir)
    ordered = _joints_root_to_tip(root)
    for joint in root.findall("joint"):
        root.remove(joint)
    root.extend(ordered)
    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


def _root_link_name(robot: ET.Element) -> str:
    links = [link.get("name") for link in robot.findall("link")]
    children = {joint.find("child").get("link") for joint in robot.findall("joint")}
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise ValueError(f"URDF の root link が 1 つに定まらない: {roots}")
    return roots[0]


def _joints_root_to_tip(robot: ET.Element) -> list[ET.Element]:
    """root link から深さ優先でたどった順の joint。兄弟は元の並び順を保つ。"""
    by_parent: dict[str, list[ET.Element]] = defaultdict(list)
    for joint in robot.findall("joint"):
        by_parent[joint.find("parent").get("link")].append(joint)

    ordered: list[ET.Element] = []
    stack = [_root_link_name(robot)]
    while stack:
        link = stack.pop()
        children = by_parent.get(link, [])
        ordered.extend(children)
        # 逆順に積むと、先頭の子から先にたどる
        stack.extend(joint.find("child").get("link") for joint in reversed(children))
    return ordered


def _absolutize_mesh_paths(root: ET.Element, mesh_dir: str) -> None:
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename is None or "://" in filename or os.path.isabs(filename):
            continue
        mesh.set("filename", os.path.abspath(os.path.join(mesh_dir, filename)))
