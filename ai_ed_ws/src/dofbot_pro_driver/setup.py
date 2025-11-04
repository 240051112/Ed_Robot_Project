from setuptools import setup
import os
from glob import glob

package_name = 'dofbot_pro_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='agboredouard51@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'arm_driver = dofbot_pro_driver.arm_driver:main',
            'apriltag_detect = dofbot_pro_driver.apriltag_detect:main',
            'grasp = dofbot_pro_driver.grasp:main',
            'calculate_volume = dofbot_pro_driver.calculate_volume:main',
            'dofbot_pro_driver = dofbot_pro_driver.dofbot_pro_driver:main',
            'test = dofbot_pro_driver.test:main',
            'apriltag_list = dofbot_pro_driver.apriltag_list:main',
            'apriltag_remove_higher = dofbot_pro_driver.apriltag_remove_higher:main'
        ],
    },
)