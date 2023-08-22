#ifndef NET_IMPL_H
#define NET_IMPL_H

#include <stdio.h>
#include <vector>
#include "wavernn.h"

const int RES_BLOCKS = 10;
const int UPSAMPLE_LAYERS = 3;

Vectorf softmax( const Vectorf& x );

class ResBlock{
    std::vector<TorchLayer> resblock;
public:
    ResBlock() = default;
    void loadNext( FILE* fd );
    Matrixf apply( const Matrixf& x );
};

class Resnet{
    TorchLayer conv_in;
    TorchLayer batch_norm;
    ResBlock resblock;
    TorchLayer conv_out;
    TorchLayer stretch2d;  //moved stretch2d layer into resnet from upsample as in python code

public:
    Resnet() = default;
    void loadNext( FILE* fd );
    Matrixf apply( const Matrixf& x );
};

class UpsampleNetwork{
    std::vector<TorchLayer> up_layers;

public:
    UpsampleNetwork() = default;
    void loadNext( FILE* fd );
    Matrixf apply( const Matrixf& x );
};

class Model{

    struct  Header{
        int num_res_blocks;
        int num_upsample;
        int total_scale;
        int nPad;
    };

    Header header;

    // Sub Networks
    UpsampleNetwork upsample;
    Resnet resnet;
    TorchLayer I;
    TorchLayer rnn1;
    TorchLayer rnn2;
    TorchLayer rnn3;
    TorchLayer rnn4;
    TorchLayer fc1;
    TorchLayer fc2;
    TorchLayer fc3;
    TorchLayer fc4;
    TorchLayer fc5;

public:
    Model() = default;
    void loadNext( FILE* fd );
    Vectorf apply( const Matrixf& x );
};


#endif // NET_IMPL_H
