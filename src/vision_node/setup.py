from setuptools import find_packages, setup

package_name = 'vision_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='junh001224@gmail.com',
    description='YOLO 기반 신발 검사 vision 노드',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vision_node = vision_node.vision_node:main',
            'capture_cameras = vision_node.capture_cameras:main',
        ],
    },
)
