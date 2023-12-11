##  Copyright (c) 2018-2023, Carnegie Mellon University
##  All rights reserved.
##
##  See LICENSE file for full information
##  SPDX-License-Identifier: BSD-3-Clause

##  fftx/setup.py

from setuptools import setup, find_packages
import codecs
import os.path
import glob

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

def find_data_files():
    examples_dir = 'examples'
    example_files = glob.glob(os.path.join(examples_dir, '*.py'))
    example_files = example_files + glob.glob(os.path.join(examples_dir, 'test-many.sh'))

    misc_dir = '.'
    misc_files = glob.glob(os.path.join(misc_dir, 'Contributing.md'))
    misc_files = misc_files + glob.glob(os.path.join(misc_dir, 'README.md'))

    return [('share/fftx/examples', example_files)] + [('share/fftx', misc_files)]

setup(
    name="fftx",
    version=get_version("fftx/__init__.py"),
    license="BSD-3-Clause",
    description="A Python front end to access the FFTX Project",
    long_description="This Python package provides NumPy/CuPy-compatible access to the high performance multi-platform code generation capabilities of the FFTX Project (https://github.com/spiral-software/fftx).  Both FFTX and this Python package originated under the Exascale Computing Project, with a focus on perforance-portable FFTs and FFT-using methods for the rapidly evolving high performance computing landscape.",
    url="https://github.com/spiral-software/python-package-fftx",
    project_urls={
        "Source Code": "https://github.com/spiral-software/python-package-fftx",
    },
    author="SpiralGen Inc.",
    author_email="Patrick.Broderick@spiralgen.com, Mike.Franusich@spiralgen.com",
    maintainer=["Patrick Broderick", "Mike Franusich"],
    maintainer_email="Patrick.Broderick@spiralgen.com, Mike.Franusich@spiralgen.com",
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    data_files=find_data_files(),
    install_requires=['spiralpy'],
)
