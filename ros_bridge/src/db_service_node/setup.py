from setuptools import setup

package_name = 'db_service_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'pymongo',
        'python-dotenv'
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='roeezach15@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'db_service = db_service_node.db_service:main',
        ],
    },
)