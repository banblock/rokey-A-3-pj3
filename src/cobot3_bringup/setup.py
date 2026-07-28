import os
from glob import glob

from setuptools import setup


package_name = "cobot3_bringup"


setup(
    name=package_name,
    version="0.1.0",
    packages=[],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rokey",
    maintainer_email="junh001224@gmail.com",
    description="Launch package for the complete shoe recycling system.",
    license="TODO: License declaration",
)
