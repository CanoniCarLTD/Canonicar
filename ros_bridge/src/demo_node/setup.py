from setuptools import setup

package_name = 'demo_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='etai',
    maintainer_email='etai444@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher_node = demo_node.publisher_node:main',
            'ppo_node = demo_node.ppo_node:main',
            'demo_data = demo_node.demo_data:main',
        ],
    },
)