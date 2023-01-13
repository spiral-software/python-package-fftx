Python Package for FFTX
=======================

This Python package provides NumPy/CuPy-compatible access to the high performance multi-platform 
code generation capabilities of the [FFTX Project](https://github.com/spiral-software/fftx), 
which is currently under development as part of the Department of Energy's 
[Exascale Computing Project](https://www.exascaleproject.org/).

The package provides a single API for several FFTs, along with some convolution-like transforms, that run on either CPUs or GPUs (NVIDIA and AMD) and supports single and double precision and both C (row-major) and Fortran (column-major) arrays.  The API uses the dimensions, datatype, ordering, and location of an array to determine which variant of a transform to invoke, simplifying application code intended for multiple target environments.

The package has successfully run on a wide range of platforms from the Raspberry Pi through some of the latest supercomputers, and is in use on personal computers running Windows, Linux, and MacOS.

## Package Contents

The FFTX Python package has two modules:

- **fft**

   Direct replacements for several NumPy/CuPy fft functions.
   
- **convo**

   Convolution and convolultion-like functions.

See the docstrings in the code or use Python's `help()` function to view a module's internal documentation.  For example:

```python console
>>> import fftx
>>> help(fftx.fft)
```

## Installation

The FFTX Python package depends on the [SnowWhite](https://github.com/spiral-software/python-package-snowwhite) package. 
Follow the instructions in SnowWhite's 
[README](https://github.com/spiral-software/python-package-snowwhite#readme)
to install and 
configure SnowWhite and its dependencies. Once you have SnowWhite working properly 
clone **python-package-fftx** into the same root directory as SnowWhite and rename it **fftx**.

## Examples

For a quick start, look through the example scripts in the **examples** subdirectory.  Run an example script without arguments to see its help message.  For example:

```shell
python run-fftn.py
usage: run-fftn sz [ F|I [ d|s [ GPU|CPU ]]]
  sz is N or N1,N2,N3
  F  = Forward, I = Inverse           (default: Forward)
  d  = double, s = single precision   (default: double precision)

  (GPU is default target unless none exists or no CuPy)

Three-dimensional complex FFT
```

Typically the examples run an FFTX function along with the equivalent NumPy/CuPy function and compare the results.  Notice how the FFTX functions support both NumPy and CuPy arrays transparently.

The first time you run a script for a specific set of arguments, you will see a a few extra lines of output from about code generation, but on subsequent runs you will not, as the generated libraries 
are cached in the SnowWhite **.libs** subdirectory.
