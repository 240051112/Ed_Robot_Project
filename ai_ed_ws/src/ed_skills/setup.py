from setuptools import setup
import os
from glob import glob

package_name = 'ed_skills'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name], # Use the explicit package name
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description='The skill server for the Ed robot.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'skill_server = ed_skills.skill_server:main',
        ],
    },
)