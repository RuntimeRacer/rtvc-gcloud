/* Copyright 2019 Eugene Ingerman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <stdio.h>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>

#include "wavernn.h"
#include "net_impl.h"

Vectorf softmax( const Vectorf& x )
{
    float maxVal = x.maxCoeff();
    Vectorf y = x.array()-maxVal;

    y = Eigen::exp(y.array());
    float sum = y.sum();
    return y.array() / sum;
}


void ResBlock::loadNext(FILE *fd)
{
    resblock.resize( RES_BLOCKS*4 );
    for(int i=0; i<RES_BLOCKS*4; ++i){
        resblock[i].loadNext(fd);
    }
}

Matrixf ResBlock::apply(const Matrixf &x)
{
    Matrixf y = x;

    for(int i=0; i<RES_BLOCKS; ++i){
        Matrixf residual = y;

        y = resblock[4*i](y);    //conv1
        y = resblock[4*i+1](y);  //batch_norm1
        y = relu(y);
        y = resblock[4*i+2](y);  //conv2
        y = resblock[4*i+3](y);  //batch_norm2

        y += residual;
    }
    return y;
}

void Resnet::loadNext(FILE *fd)
{
    conv_in.loadNext(fd);
    batch_norm.loadNext(fd);
    resblock.loadNext(fd); //load the full stack
    conv_out.loadNext(fd);
    stretch2d.loadNext(fd);
}

Matrixf Resnet::apply(const Matrixf &x)
{
    Matrixf y = x;
    y=conv_in(y);
    y=batch_norm(y);
    y=relu(y);
    y=resblock.apply(y);
    y=conv_out(y);
    y=stretch2d(y);
    return y;
}

void UpsampleNetwork::loadNext(FILE *fd)
{
    up_layers.resize( UPSAMPLE_LAYERS*2 );
    for(int i=0; i<up_layers.size(); ++i){
        up_layers[i].loadNext(fd);
    }
}

Matrixf UpsampleNetwork::apply(const Matrixf &x)
{
    Matrixf y = x;
    for(int i=0; i<up_layers.size(); ++i){
        y = up_layers[i].apply( y );
    }
    return y;
}

void Model::loadNext(FILE *fd)
{
    fread( &header, sizeof( Model::Header ), 1, fd);

    resnet.loadNext(fd);
    upsample.loadNext(fd);

    I.loadNext(fd);
    rnn1.loadNext(fd);
    fc1.loadNext(fd);
    fc2.loadNext(fd);
}


Matrixf pad( const Matrixf& x, int nPad )
{
    Matrixf y = Matrixf::Zero(x.rows(), x.cols()+2*nPad);
    y.block(0, nPad, x.rows(), x.cols() ) = x;
    return y;
}

Vectorf vstack( const Vectorf& x1, const Vectorf& x2 )
{
    Vectorf temp(x1.size()+x2.size());
    temp << x1, x2;
    return temp;
}

Vectorf vstack( const Vectorf& x1, const Vectorf& x2, const Vectorf& x3 )
{
    return vstack( vstack( x1, x2), x3 );
}

inline float sampleCategorical( const VectorXf& probabilities )
{
    //Sampling using this algorithm https://en.wikipedia.org/wiki/Categorical_distribution#Sampling
    static std::ranlux24 rnd;
    std::vector<float> cdf(probabilities.size());
    float uniform_random = static_cast<float>(rnd()) / rnd.max();

    std::partial_sum(probabilities.data(), probabilities.data()+probabilities.size(), cdf.begin());
    auto it = std::find_if(cdf.cbegin(), cdf.cend(), [uniform_random](float x){ return (x >= uniform_random);});
    int pos = std::distance(cdf.cbegin(), it);
    return pos;
}

inline float invMulawQuantize( float x_mu )
{
    const float mu = MULAW_QUANTIZE_CHANNELS - 1;
    float x = (x_mu / mu) * 2.f - 1.f;
    x = std::copysign(1.f, x) * (std::exp(std::fabs(x) * std::log1p(mu) ) - 1.f) / mu;
    return x;
}

Vectorf Model::apply(const Matrixf &mels_in)
{
    std::vector<int> rnn_shape = rnn1.shape();

    Matrixf mel_padded = pad(mels_in, header.nPad);
    Matrixf mels = upsample.apply(mel_padded);
    int indent = header.nPad * header.total_scale;

    mels = mels.block(0,indent, mels.rows(), mels.cols()-2*indent ).eval(); //remove padding added in the previous step

    Matrixf aux = resnet.apply(mel_padded);

    assert(mels.cols() == aux.cols());
    int seq_len = mels.cols();
    int n_aux = aux.rows();

    Matrixf a1 = aux.block(0, 0, n_aux/2-1, aux.cols()); //we are throwing away the last aux row to keep network input a mulitple of 8.
    Matrixf a2 = aux.block(n_aux/2, 0, n_aux/2, aux.cols());

    Vectorf wav_out(seq_len);     //output vector

    Vectorf x = Vectorf::Zero(1); //current sound amplitude

    Vectorf h1 = Vectorf::Zero(rnn_shape[0]);

    for(int i=0; i<seq_len; ++i){
        Vectorf y = vstack( x, mels.col(i), a1.col(i) );
        y = I( y );
        h1 = rnn1( y, h1 );
        y += h1;

        y = vstack( y, a2.col(i) );

        y = relu( fc1( y ) );
        Vectorf logits = fc2( y );
        Vectorf posterior = softmax( logits );

        float newAmplitude = sampleCategorical( posterior );
        newAmplitude = (2.*newAmplitude) / (posterior.size()-1.) - 1.; //for bits output
        //newAmplitude = invMulawQuantize( newAmplitude );   //mulaw output
        wav_out(i) = x(0) = newAmplitude;

    }

    return wav_out;
}


