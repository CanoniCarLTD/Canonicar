from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'map_loader'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'map_loader'), glob('map_loader/*.xodr')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kfirh',
    maintainer_email='kfir.hoftman@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'load_map_node = map_loader.load_map:main',
        ],
    },
)
