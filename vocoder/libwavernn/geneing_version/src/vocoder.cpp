/*
Copyright 2019 Eugene Ingerman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <stdio.h>
#include <string>
#include "cxxopts.hpp"

#include "cnpy.h"
#include "vocoder.h"
#include "net_impl.h"
#include "wavernn.h"

using namespace std;

Matrixf loadMel( string npy_fname )
{
    // Fixed With peeking at: https://rancheng.github.io/npy-cpp/
    cnpy::NpyArray npy_data = cnpy::npy_load(npy_fname);
    int nRows = npy_data.shape[0];
    int nCols = npy_data.shape[1];

    Matrixf mel( nRows, nCols );
    memcpy(mel.data(), npy_data.data<float>(), nRows * nCols * sizeof(float));

    // Important!
    mel.transposeInPlace();

    // Fixme: Apply max_abs_value similar to NumPy Operation

    return mel;
}

int main(int argc, char* argv[])
{

    cxxopts::Options options("vocoder", "WaveRNN based vocoder");
    options.add_options()
            ("w,weights", "File with network weights", cxxopts::value<string>()->default_value(""))
            ("m,mel", "File with mel inputs", cxxopts::value<string>()->default_value(""))
            ;
    auto result = options.parse(argc, argv);

    string weights_file = result["weights"].as<string>();
    string mel_file = result["mel"].as<string>();

    Matrixf mel = loadMel( mel_file );

    FILE *fd = fopen(weights_file.c_str(), "rb");
    assert(fd);

    Model model;
    model.loadNext(fd);
    fclose(fd);

    Vectorf wav = model.apply(mel);

    // Fixme: Proper MuLaw-Decode using Matrix Operation similar to NumPy

    // Fixme: Save as .wav instead of binary
    FILE *fdout = fopen("wavout.bin","wb");
    fwrite(wav.data(), sizeof(float), wav.size(), fdout);
    fclose(fdout);

//    TorchLayer I;  I.loadNext(fd);
//    TorchLayer GRU; GRU.loadNext(fd);
//    TorchLayer conv_in; conv_in.loadNext(fd);
//    TorchLayer conv_1; conv_1.loadNext(fd);
//    TorchLayer conv_2d; conv_2d.loadNext(fd);
//    TorchLayer batch_norm; batch_norm.loadNext(fd);

// Test for linear layer
//    Vectorf x(112);
//    for(int j=1; j<=112; ++j)
//        x(j-1) = 1. + 1./j;
//    Vectorf x1, x2;
//    x1 = I(x);


//    Vectorf x(512), hx(512);

//    for(int j=1; j<=512; ++j){
//        x(j-1) = 1. + 1./j;
//        hx(j-1) = -3. + 2./j;
//    }

//    Vectorf h1 = GRU(x, hx);

//    Matrixf mel(128,10);
//    for(int i=0; i<mel.rows(); ++i){
//        for(int j=0; j<mel.cols(); ++j){
//            mel(i,j) = (1.+1./(i+1))*(-3.+2./(j+1));
//        }
//    }

    //Matrixf aux = conv_in(mel);
    //Matrixf cv2 = conv_2d(mel);
//    Matrixf batchnorm = batch_norm(mel);

    fclose(fd);
    fclose(fdMel);

    return 0;
}
