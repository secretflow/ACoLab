from setuptools import find_packages, setup

with open("../README.md", "r") as f:
    long_description = f.read()

setup(
    name='privacy_utility_framework',
    version='0.0.10',
    author='KS',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[],
)