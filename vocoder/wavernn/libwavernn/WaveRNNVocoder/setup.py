# Easy install of C++ Vocoder Module via PIP:
#
# Copy over generated .so file
#
# - for install:
#      1. copy over .so file in this dir and rename it to WaveRNNVocoder.so
#      2. cd into this folder
#      3. run pip install .
#
# - for uninstall, run: pip uninstall WaveRNNVocoder
#

from distutils.core import setup
setup (name = 'WaveRNNVocoder',
       version = '0.1',
       author = "RuntimeRacer",
       description = "C++ WaveRNNVocoder for Python.",
       packages=[''],
       package_data={'': ['WaveRNNVocoder.so']},
       )