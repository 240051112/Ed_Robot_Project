from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'ed_vision'

setup(
    name=package_name,
    version='0.0.2',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        # if you add launch files later, they’ll be picked up automatically
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # if you add config files later, uncomment:
        # (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description="Ed's perception node V2 (YOLO + depth → 3D detections, stabilized output)",
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'perception_node = ed_vision.perception_node:main',
        ],
    },
)
