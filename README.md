Python Module for FFTX
======================

This Python module provides access to the high performance multi-platform 
code generation capabilities of the [FFTX Project](https://github.com/spiral-software/fftx), 
which is currently under development as part of the Department of Energy's 
[Exascale Computing Project](https://www.exascaleproject.org/).

# Installation

The FFTX Python module depends on the [SnowWhite](https://github.com/spiral-software/python-package-snowwhite) module. 
Follow the instructions in SnowWhite's 
[README](https://github.com/spiral-software/python-package-snowwhite#readme)
to install and 
configure SnowWhite and its dependencies. Once you have SnowWhite working properly 
clone this module into the same root directory as SnowWhite and name it **fftx**.

# Use

This module currently offers direct replacements for the following NumPy and CuPy functions:

* fft.fft()
* fft.ifft()
* fft.fftn()
* fft.ifftn()
* fft.rfftn()
* fft.irfftn()

To use an fftx version of one of these Numpy functions, import the fftx module and change 
from numpy to fftx in the function's full name, for example:

```
import numpy as np
import fftx
.
.
.
oldway = np.fft.fftn(src)
newway = fftx.fft.fftn(src)

```

The fftn() and ifftn() functions have an option for generating CUDA code.
See **run-fftn.py** in the **examples** subdirectory.

# Running Examples

To run a script from the **examples** subdirectory, copy the script into a scratch directory
and run it with:

```python <script name>```

Note that on some systems you have to specify **python3** instead of just **python**.

Edit your copy of the script to change arguments, as noted in the scripts, to try different
variations of the transforms.

The first time you run a script for a specific set of arguments, you will see a lot of output generated during code generation, but on subsequent runs you will not, as the generated libraries 
are cached in the SnowWhite **.libs** subdirectory.



