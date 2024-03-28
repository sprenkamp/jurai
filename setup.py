from setuptools import setup, find_packages

setup(
    name='jurai',
    version='0.1',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    install_requires=[
        # list your dependencies here
    ],
)