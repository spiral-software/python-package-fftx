Python Package for FFTX
=======================

This Python package provides NumPy/CuPy-compatible access to the high performance multi-platform 
code generation capabilities of the [FFTX Project](https://github.com/spiral-software/fftx).  Both
FFTX and this Python package originated as part of the Department of Energy's [Exascale Computing
Project](https://www.exascaleproject.org/). 

The package provides a single API for several FFTs, along with some convolution-like transforms,
that run on either CPUs or GPUs (NVIDIA and AMD) and supports single and double precision and both
C (row-major) and Fortran (column-major) arrays.  The API uses the dimensions, datatype, ordering,
and location of an array to determine which variant of a transform to invoke, simplifying
application code intended for multiple target environments.

The package has successfully run on a wide range of platforms from the Raspberry Pi through some
of the latest supercomputers, and is in use on personal computers running Windows, Linux, and
MacOS.

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

The easiest method to install **fftx** is to use Python's package manager **pip**, e.g.,
```shell
pip install fftx
```

When you install **fftx** the modules are installed in your ```site-packages``` location.  You
can determine this location (and your base location) by running this python command:
```shell
python -m site --user-site --user-base
##   Sample output -- On Unix/Linux/MacOS
##   ~/.local:~/.local/lib/python3.9/site-packages
##   Windows
##   C:\Users\<user>\AppData\Roaming\Python;C:\Users\<user>\AppData\Roaming\Python\Python38\site-packages
```

Base location is always output first, the two locations are separated by a ':' (Unix/Linux/MacOS)
or a ';' (Windows).

The examples and other miscellaneous files will be installed at ```share/fftx``` under your
base location.

If you want to develop, contribute to, or possibly modify **fftx** it may be better to clone
or fork the **fftx** repository, see
[**Contributing**](https://github.com/spiral-software/python-package-fftx/blob/main/Contributing.md).
Clone **python-package-fftx** to a location on your computer renaming it to **fftx**.  For
example:
```shell
cd ~/work/python-packages
git clone https://github.com/spiral-software/python-package-fftx fftx
```

If you clone or fork the **fftx** repository (vs installing the package) you need to add the
directory that contains it to the environment variable **PYTHONPATH**.  (In the above example that
would be ```~/work/python-packages/fftx```).  This allows Python to locate the **fftx**
module.

NOTE:  The **fftx** Python package depends on the
[SpiralPy](https://github.com/spiral-software/python-package-spiralpy) package.  If you install
using **pip** that module will be automatically installed also.  However, if you install maually,
you will need to download and install **SpiralPy** also.  Follow the instructions in **SpiralPy**'s 
[README](https://github.com/spiral-software/python-package-spiralpy#readme) to install and 
configure **SpiralPy** and its dependencies. Once you have **SpiralPy** working properly 
clone **python-package-fftx** into the same root directory as **SpiralPy** and rename it **fftx**.

## Examples

For a quick start, look through the example scripts in the **examples** subdirectory.  Run an
example script without arguments to see its help message.  For example:

```shell
python run-fftn.py
usage: run-fftn sz [ F|I [ d|s [ GPU|CPU ]]]
  sz is N or N1,N2,.. all N >= 2, single N implies 3D cube
  F  = Forward, I = Inverse           (default: Forward)
  d  = double, s = single precision   (default: double precision)
                                    
  (GPU is default target unless none exists or no CuPy)
  
Multi-dimensional complex FFT
```

Typically the examples run an FFTX function along with the equivalent NumPy/CuPy function and
compare the results.  Notice how the FFTX functions support both NumPy and CuPy arrays
transparently. 

The first time you run a script for a specific set of arguments, you will see a a few extra lines
of output from about code generation, but on subsequent runs you will not, as the generated
libraries are cached as explained in the **SpiralPy**
[README](https://github.com/spiral-software/python-package-spiralpy#readme).
