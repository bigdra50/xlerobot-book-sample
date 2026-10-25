"""公式の XLeRobot の URDF（Vector-Wangel/XLeRobot の ManiSkill 用）から、台車と腕と head の xacro のマクロを作る。

    python3 urdf/make_official_body.py <XLeRobot のリポジトリ>

シミュレーター用に台車を平面で動かす仮想の関節（root_x_axis_joint、root_y_axis_joint、root_z_rotation_joint）と
そのリンクを外し、base_link を根にする。メッシュは参照されているものだけを urdf/official/meshes へ写し、
パスを package://xlerobot_nav/urdf/official/meshes/ に書き換える。
公式の base_link は床の上の台車の中心である。マクロは、この点を引数 parent のリンクに取り付ける。
回転の中心（この台車の base_link）から台車の中心への変換は、呼び出す側の URDF が決める。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "official"
VIRTUAL_JOINTS = {"root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"}
VIRTUAL_LINKS = {"root", "root_arm_1_link_1", "root_arm_1_link_2", "base_link"}


def main() -> None:
    repo = Path(sys.argv[1]).resolve()
    src_dir = repo / "simulation" / "Maniskill" / "assets" / "xlerobot"
    root = ET.parse(src_dir / "xlerobot.urdf").getroot()
    commit = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "log",
            "-1",
            "--format=%h %ad",
            "--date=short",
            "--",
            str(src_dir / "xlerobot.urdf"),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    body = ET.Element("xacro:macro", {"name": "xlerobot_body", "params": "parent"})
    meshes: set[str] = set()
    for el in root:
        if el.tag == "joint" and el.get("name") in VIRTUAL_JOINTS:
            continue
        # base_link は、呼び出す側の URDF にすでにある。見た目（台車のメッシュ）だけは残す
        if el.tag == "link" and el.get("name") in VIRTUAL_LINKS - {"base_link"}:
            continue
        if el.tag == "link" and el.get("name") == "base_link":
            el = _base_visual(el)
        for m in el.iter("mesh"):
            name = Path(m.get("filename")).name
            meshes.add(name)
            m.set("filename", f"package://xlerobot_nav/urdf/official/meshes/{name}")
        for parent in el.iter("parent"):
            if parent.get("link") == "base_link":
                parent.set("link", "${parent}")
        body.append(el)
    body.append(
        ET.fromstring(
            '<joint name="xlerobot_cart_joint" type="fixed"><parent link="${parent}"/><child link="xlerobot_cart"/></joint>'
        )
    )

    OUT.mkdir(exist_ok=True)
    (OUT / "meshes").mkdir(exist_ok=True)
    for name in sorted(meshes):
        shutil.copy2(src_dir / "meshes" / name, OUT / "meshes" / name)
    shutil.copy2(repo / "LICENSE", OUT / "LICENSE")

    robot = ET.Element("robot", {"xmlns:xacro": "http://www.ros.org/wiki/xacro"})
    robot.append(
        ET.Comment(
            f" 生成物。make_official_body.py で作る。元は Vector-Wangel/XLeRobot {commit} の "
            "simulation/Maniskill/assets/xlerobot/xlerobot.urdf（Apache-2.0） "
        )
    )
    robot.append(body)
    ET.indent(robot)
    (OUT / "xlerobot_body.xacro").write_text(
        '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode") + "\n"
    )
    print(f"meshes {len(meshes)}, from {commit}")


def _base_visual(link: ET.Element) -> ET.Element:
    """公式の base_link の見た目と当たり判定を、base_link に固定した別のリンクへ移す。"""
    new = ET.Element("link", {"name": "xlerobot_cart"})
    for child in link:
        new.append(child)
    return new


if __name__ == "__main__":
    main()
