from setuptools import setup

package_name = 'hello_world'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    entry_points={
        'console_scripts': [
            'publisher = hello_world.publisher:main',
            'subscriber = hello_world.subscriber:main',
        ],
    },
)
