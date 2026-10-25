import os
from glob import glob

from setuptools import find_packages, setup

package_name = "xlerobot_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*.xacro")),
        (
            os.path.join("share", package_name, "urdf", "official"),
            glob("urdf/official/*.xacro") + ["urdf/official/LICENSE"],
        ),
        (os.path.join("share", package_name, "urdf", "official", "meshes"), glob("urdf/official/meshes/*")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml") + glob("config/*.xml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="bigdra50",
    maintainer_email="bigdra50@users.noreply.github.com",
    description="XLeRobot base bridge and wheel/ARKit odometry fusion.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "base_bridge = xlerobot_nav.base_bridge:main",
            "odom_fusion = xlerobot_nav.odom_fusion:main",
            "drive_pattern = xlerobot_nav.drive_pattern:main",
            "device_params = xlerobot_nav.device_params:main",
            "table_dock = xlerobot_nav.table_dock:main",
            "saved_map = xlerobot_nav.saved_map:main",
            "patrol = xlerobot_nav.patrol:main",
            "host_bridge = xlerobot_nav.host_bridge:main",
            "goto_place = xlerobot_nav.goto_place:main",
        ],
    },
)
