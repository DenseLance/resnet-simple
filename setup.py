#!/usr/bin/python3
import setuptools

# METADATA
NAME = "resnet-simple"
VERSION = "1.0.1"
AUTHOR = "Lance Chin"
EMAIL = "denselance@gmail.com"
DESCRIPTION = "Python package that provides a well-documented and easy to use implementation of ResNet (and ResNetv1.5), together with its most basic use case of image classification."
URL = "https://github.com/DenseLance/resnet-simple"
REQUIRES_PYTHON = ">=3.7.0"

with open("requirements.txt", "r") as f:
    DEPENDENCIES = f.read().splitlines()
    f.close()

with open("README.md", "r", encoding = "utf-8") as f:
    LONG_DESCRIPTION = f.read()
    f.close()

setuptools.setup(
    name = NAME,
    version = VERSION,
    author = AUTHOR,
    author_email = EMAIL,
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    long_description_content_type = "text/markdown",
    url = URL,
    project_urls={
        "Bug Tracker": "https://github.com/DenseLance/resnet-simple/issues",
    },
    license = "MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ],
    packages = ["resnet_simple", "resnet_simple.resnet"],
    python_requires = REQUIRES_PYTHON,
    install_requires = DEPENDENCIES
)
