#!/bin/bash

# IMPORTANT! - WaveRNN Module variant to build
BUILD_VARIANT="fatchord_version"

# Uninstall any old version of the lib
# pip uninstall WaveRNNVocoder

# Clean dirs & Build the Library
rm -rf build
mkdir -p build
cd build || (print "unable to create build dir" && exit 1)

# build using standalone Python env
cmake ../$BUILD_VARIANT/src
make

# => This creates a couple python modules in /build.
# Copy the WaveRNNVocoder.so into the 'WaveRNNVocoder' folder and install / uninstall via pip:
cd ..
cp build/WaveRNNVocoder*.so WaveRNNVocoder/WaveRNNVocoder.so
pip install WaveRNNVocoder/.

