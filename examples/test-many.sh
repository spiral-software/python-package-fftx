#! /bin/bash

# Copyright 2018-2023, Carnegie Mellon University
# All rights reserved.
#
# See LICENSE (https://github.com/spiral-software/python-package-fftx/blob/main/LICENSE)

##  run a couple of samples for each example script
##  if FFTX_HOME points to a location with prebuilt libraries then we'll exercise library code
##  as well as building some on the fly

unset SP_LIBRARY_PATH

echo "Run run-fftn.py 64"
python run-fftn.py 64
echo "Run run-fftn.py 24,36,42"
python run-fftn.py 24,36,42
echo "Run run-fftn.py 100"
python run-fftn.py 100

export SP_LIBRARY_PATH=$FFTX_HOME/lib

echo "Run run-fftn.py 100,224,224 F"
python run-fftn.py 100,224,224 F
echo "Run run-fftn.py 100,224,224 I"
python run-fftn.py 100,224,224 I

echo "Run run-mdrconv.py 64"
python run-mdrconv.py 64
echo "Run run-mdrconv.py 64,64,96"
python run-mdrconv.py 64,64,96

echo "Run run-mdrfsconv.py 32"
python run-mdrfsconv.py 32
echo "Run run-mdrfsconv.py 64"
python run-mdrfsconv.py 64
echo "Run run-mdrfsconv.py 96"
python run-mdrfsconv.py 96

echo "Run run-rfftn.py 80"
python run-rfftn.py 80
echo "Run run-rfftn.py 100"
python run-rfftn.py 100
echo "Run run-rfftn.py 80 I"
python run-rfftn.py 80 I
echo "Run run-rfftn.py 100 I"
python run-rfftn.py 100 I
echo "Run run-rfftn.py 100,224,224"
python run-rfftn.py 100,224,224
echo "Run run-rfftn.py 224,224,100"
python run-rfftn.py 224,224,100
echo "Run run-rfftn.py 224,224,120"
python run-rfftn.py 224,224,120

echo "Run run-stepphase.py 81"
python run-stepphase.py 81
echo "Run run-stepphase.py 125"
python run-stepphase.py 125
echo "Run run-stepphase.py 225"
python run-stepphase.py 225

echo "Run time-fftn.py 100"
python time-fftn.py 100
echo "Run time-fftn.py 100,224,224"
python time-fftn.py 100,224,224

exit 0
