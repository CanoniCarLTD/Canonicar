from setuptools import find_packages, setup

package_name = 'client_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/client_node',
             ['client_node/sensors_config.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kfir',
    maintainer_email='kfir.hoftman@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spawn_vehicle_node = client_node.spawn_vehicle:main',
        ],
    },
)
