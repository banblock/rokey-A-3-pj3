import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'sorting_line_fms'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # fleet_config.py는 isaacpjt/sorting_line/에 있는 순수 데이터 모듈(rclpy 의존
        # 없음, Isaac Sim 스크립트도 그대로 가져다 씀) — 패키지 안으로 옮기지 않고
        # share/에 설치해서 ament_index로 찾아 쓴다(설치 방식과 무관하게 항상 동작).
        ('share/' + package_name, ['../../isaacpjt/sorting_line/fleet_config.py']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='junh001224@gmail.com',
    description='신발 분류 소팅라인 구동 노드 — FMS(관제탑)/Fleet Driver(중간관리자)',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'fms_node = sorting_line_fms.fms_node:main',
            'fleet_driver = sorting_line_fms.fleet_driver:main',
            'main_control_stub = sorting_line_fms.main_control_stub:main',
        ],
    },
)
