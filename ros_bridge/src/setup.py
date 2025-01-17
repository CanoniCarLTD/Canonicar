from setuptools import setup

package_name = 'my_hello_world'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Hello World ROS 2 package',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher = my_hello_world.publisher:main',
            'subscriber = my_hello_world.subscriber:main',
        ],
    },
)
