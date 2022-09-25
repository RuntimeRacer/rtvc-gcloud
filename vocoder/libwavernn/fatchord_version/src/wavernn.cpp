/*
Copyright 2019 Eugene Ingerman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <stdio.h>
#include <iostream>
#include <cmath>
#include "wavernn.h"


Matrixf relu( const Matrixf& x){
    return x.array().max(0.f);
    //return x.unaryExpr([](float x){return std::max(0.f, x);});
}

inline Vectorf sigmoid( const Vectorf& v )
{
    //TODO: optimize this
    //maybe use one of these approximations: https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm
    Vectorf y = 1.f / ( 1.f + Eigen::exp( - v.array()));
    return y;
}

inline Vectorf tanh( const Vectorf& v )
{
    //TODO: optimize this
    Vectorf y = Eigen::tanh( v.array() );
    return y;
}

BaseLayer *TorchLayer::loadNext(FILE *fd)
{
    TorchLayer::Header header;
    fread(&header, sizeof(TorchLayer::Header), 1, fd);

    std::cerr << "Loading:" << header.name << std::endl;

    switch( header.layerType ){

    case TorchLayer::Header::LayerType::Linear:
    {
        impl = new LinearLayer();
        impl->loadNext(fd);
        return impl;
    }
    break;

    case TorchLayer::Header::LayerType::GRU:
    {
        impl = new GRULayer();
        impl->loadNext(fd);
        return impl;
    }
    break;

    case TorchLayer::Header::LayerType::Conv1d:
    {
        impl = new Conv1dLayer();
        impl->loadNext(fd);
        return impl;
    }
    case TorchLayer::Header::LayerType::Conv2d:{
        impl = new Conv2dLayer();
        impl->loadNext(fd);
        return impl;
    }
    case TorchLayer::Header::LayerType::BatchNorm1d:
    {
        impl = new BatchNorm1dLayer();
        impl->loadNext(fd);
        return impl;
    }
    case TorchLayer::Header::LayerType::Stretch2d:
    {
        impl = new Stretch2dLayer();
        impl->loadNext(fd);
        return impl;
    }

    default:
        return nullptr;
    }
}


LinearLayer* LinearLayer::loadNext(FILE *fd)
{
    LinearLayer::Header header;
    fread( &header, sizeof(LinearLayer::Header), 1, fd);
    assert(header.elSize==4 or header.elSize==2);

    mat.read(fd, header.elSize, header.nRows, header.nCols); //read compressed array

    bias.resize(header.nRows);
    fread(bias.data(), header.elSize, header.nRows, fd);
    return this;
}

Vectorf LinearLayer::apply(const Vectorf &x)
{
    return (mat*x)+bias;
}

GRULayer* GRULayer::loadNext(FILE *fd)
{
    GRULayer::Header header;
    fread( &header, sizeof(GRULayer::Header), 1, fd);
    assert(header.elSize==4 or header.elSize==2);

    nRows = header.nHidden;
    nCols = header.nInput;

    b_ir.resize(header.nHidden);
    b_iz.resize(header.nHidden);
    b_in.resize(header.nHidden);

    b_hr.resize(header.nHidden);
    b_hz.resize(header.nHidden);
    b_hn.resize(header.nHidden);


    W_ir.read( fd, header.elSize, header.nHidden, header.nInput);
    W_iz.read( fd, header.elSize, header.nHidden, header.nInput);
    W_in.read( fd, header.elSize, header.nHidden, header.nInput);

    W_hr.read( fd, header.elSize, header.nHidden, header.nHidden);
    W_hz.read( fd, header.elSize, header.nHidden, header.nHidden);
    W_hn.read( fd, header.elSize, header.nHidden, header.nHidden);

    fread(b_ir.data(), header.elSize, header.nHidden, fd);
    fread(b_iz.data(), header.elSize, header.nHidden, fd);
    fread(b_in.data(), header.elSize, header.nHidden, fd);

    fread(b_hr.data(), header.elSize, header.nHidden, fd);
    fread(b_hz.data(), header.elSize, header.nHidden, fd);
    fread(b_hn.data(), header.elSize, header.nHidden, fd);

    return this;
}


Vectorf GRULayer::apply(const Vectorf &x, const Vectorf &hx)
{
    Vectorf r, z, n, hout;

    r = sigmoid( W_ir*x + b_ir + W_hr*hx + b_hr);
    z = sigmoid( W_iz*x + b_iz + W_hz*hx + b_hz);
    n = tanh( W_in*x + b_in + (r.array() * (W_hn*hx + b_hn).array()).matrix());
    hout = (1.f-z.array()) * n.array() + z.array() * hx.array();
    return hout;
}


Vectorf CompMatrix::operator*(const Vectorf &x)
{
    Vectorf y = Vectorf::Zero(nRows);
    assert(nCols == x.size());

    int weightPos = 0;

    const float * __restrict x_ptr = x.data();
    float * __restrict y_ptr = y.data();

    for(int i=0; i<nGroups; ++i){
        float sum = 0;
        int col = SPARSE_GROUP_SIZE*colIdx[i];

        //scalar product loop. compiler should optimize it.
        for(int k=0; k<SPARSE_GROUP_SIZE; ++k){
            sum += weight[weightPos++]*x_ptr[col++];
        }
        y_ptr[rowIdx[i]] += sum;
    }

    return y;
}

Conv1dLayer *Conv1dLayer::loadNext(FILE *fd)
{
    Conv1dLayer::Header header;
    fread( &header, sizeof(Conv1dLayer::Header), 1, fd);
    assert(header.elSize==4 or header.elSize==2);
    hasBias = header.useBias;
    inChannels = header.inChannels;
    outChannels = header.outChannels;
    nKernel = header.kernelSize;

    if( nKernel==1 ){
        //if kernel is 1x then convolution is just matrix multiplication. Load weight into the first element
        //and handle separately
        weight.resize(1);
        weight[0].resize(inChannels, outChannels);
        fread(weight[0].data(), header.elSize, inChannels*outChannels*nKernel, fd);
    } else {
        weight.resize(outChannels);
        for(int i=0; i<outChannels; ++i){
            weight[i].resize(inChannels, nKernel);
            fread(weight[i].data(), header.elSize, inChannels*nKernel, fd);
        }
    }

    if( hasBias ){
        bias.resize(outChannels);
        fread(bias.data(), header.elSize, outChannels, fd);
    }
    return this;
}

Matrixf Conv1dLayer::apply(const Matrixf &x)
{
    int convDim = x.cols()-nKernel+1;
    Matrixf y(outChannels, convDim);

    if( nKernel == 1 ){
        //fast path for x1 convolution kernel
        y = weight[0] * x;
    } else {
        for(int outIdx = 0; outIdx<outChannels; ++outIdx){
            for(int kernIdx = 0; kernIdx < convDim; ++kernIdx ){
                y(outIdx, kernIdx) = ( x.block(0, kernIdx, inChannels, nKernel).cwiseProduct( weight[outIdx] ) ).sum();
            }
        }
    }

    if( hasBias ){
        //add bias to every column
        y.colwise() += bias.transpose();
    }

    return y;
}

Conv2dLayer *Conv2dLayer::loadNext(FILE *fd)
{
    Conv2dLayer::Header header;
    fread( &header, sizeof(Conv2dLayer::Header), 1, fd);
    assert(header.elSize==4 or header.elSize==2);
    nKernel = header.nKernel;

    weight.resize(nKernel);
    fread(weight.data(), header.elSize, nKernel, fd);
    return this;
}

Matrixf Conv2dLayer::apply(const Matrixf &x)
{

    Matrixf y(x.rows(), x.cols());
    int nKernel = weight.size();
    int npad = (nKernel-1)/2;

    //TODO: possibly optimize
    for(int i=0; i<x.rows(); ++i){
        Vectorf temp = Vectorf::Zero(x.cols()+2*npad);
        temp.block(0, npad, 1, x.cols()) = x.block(i, 0, 1, x.cols());
        for(int j=0; j<x.cols(); ++j){
            y(i,j) = ( temp.block(0, j, 1, nKernel).array() * weight.array() ).sum();
        }
    }

    return y;
}

BatchNorm1dLayer *BatchNorm1dLayer::loadNext(FILE *fd)
{
    BatchNorm1dLayer::Header header;
    fread( &header, sizeof( BatchNorm1dLayer::Header), 1, fd);
    assert(header.elSize==4 or header.elSize==2);

    eps = header.eps;
    nChannels = header.inChannels;

    weight.resize( header.inChannels );
    bias.resize( header.inChannels );
    running_mean.resize( header.inChannels );
    running_var.resize( header.inChannels );

    fread(weight.data(), header.elSize, header.inChannels, fd);
    fread(bias.data(), header.elSize, header.inChannels, fd);
    fread(running_mean.data(), header.elSize, header.inChannels, fd);
    fread(running_var.data(), header.elSize, header.inChannels, fd);

    return this;
}

Matrixf BatchNorm1dLayer::apply(const Matrixf &x)
{
    Matrixf y(x.rows(), x.cols());

    //y = ((x1[:,0]-running_mean)/(np.sqrt(running_var+eps)))*weight+bias

    Vectorf invstd = Eigen::rsqrt(running_var.array() + eps);
    Matrixf r1 = (x.colwise() - running_mean.transpose());
    y = ((r1.array().colwise()*invstd.transpose().array()).colwise()*weight.transpose().array()).colwise() + bias.transpose().array();
    return y;
}

Stretch2dLayer *Stretch2dLayer::loadNext(FILE *fd)
{
    Stretch2dLayer::Header header;
    fread( &header, sizeof(Stretch2dLayer::Header), 1, fd);
    x_scale = header.x_scale;
    y_scale = header.y_scale;
    return this;
}

Matrixf Stretch2dLayer::apply(const Matrixf &x)
{
    Matrixf y(x.rows()*y_scale, x.cols()*x_scale);

    assert(y_scale==1); //TODO: implement for 2d scaling
    int scaled_col = 0;
    for(int i=0; i<x.cols(); ++i){
        for(int j=0; j<x_scale; ++j){
            y.col(scaled_col++) = x.col(i);
        }
    }
    return y;
}
