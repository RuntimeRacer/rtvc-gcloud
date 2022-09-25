#ifndef WAVERNN_H
#define WAVERNN_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

const int SPARSE_GROUP_SIZE = 4; //When pruning we use groups of 4 to reduce index
const int MULAW_QUANTIZE_CHANNELS = 1024; //512 (2^9);  //same as hparams.mulaw_quantize_channels (2 ^ BITS)
const uint8_t ROW_END_MARKER = 255;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> Matrixf;
typedef Matrix<float, 1, Dynamic> Vectorf;
typedef Matrix<uint8_t, 1, Dynamic> Vectori8;


Matrixf relu( const Matrixf& x);

class CompMatrix{
    //Vectorf weight;
    //Vectori8 index;
    float __attribute__((aligned (32))) *weight;
    int __attribute__((aligned (32))) *rowIdx;
    int8_t __attribute__((aligned (32))) *colIdx; //colIdx gets multiplied by SPARSE_GROUP_SIZE to get the actual position
    int nGroups;

    int nRows, nCols;

    void prepData( std::vector<float>& wght, std::vector<uint8_t>& idx )
    {

        weight = static_cast<float*>(CompMatrix::aligned_alloc_impl(32, sizeof(float)*wght.size()));

        nGroups = wght.size()/SPARSE_GROUP_SIZE;
        rowIdx = static_cast<int*>(CompMatrix::aligned_alloc_impl(32, sizeof(int)*nGroups));
        colIdx = static_cast<int8_t*>(CompMatrix::aligned_alloc_impl(32, sizeof(int8_t)*nGroups));

        std::copy(wght.begin(), wght.end(), weight);

        int row = 0;
        int n = 0;

        for(int i=0; i<idx.size(); ++i){
            if( idx[i] == ROW_END_MARKER ){
                row++;
            } else {
                assert(n < nGroups);
                *(colIdx+n) = idx[i];
                *(rowIdx+n) = row;
                n++;
            }
        }
        assert( n == nGroups );
    };

#if _WIN32
    static void *align( std::size_t alignment, std::size_t size,
                 void *&ptr, std::size_t &space ) {
        std::uintptr_t pn = reinterpret_cast< std::uintptr_t >( ptr );
        std::uintptr_t aligned = ( pn + alignment - 1 ) & - alignment;
        std::size_t padding = aligned - pn;
        if ( space < size + padding ) return nullptr;
        space -= padding;
        return ptr = reinterpret_cast< void * >( aligned );
    }
#endif

    static void* aligned_alloc_impl(std::size_t alignment, std::size_t size){
#ifdef __linux__
        return std::aligned_alloc(alignment, size);
#elif _WIN32
        if(alignment < alignof(void*)) {
            alignment = alignof(void*);
        }
        std::size_t space = size + alignment - 1;
        void* allocated_mem = ::operator new(space + sizeof(void*));
        void* aligned_mem = static_cast<void*>(static_cast<char*>(allocated_mem) + sizeof(void*));
        ////////////// #1 ///////////////
        CompMatrix::align(alignment, size, aligned_mem, space);
        ////////////// #2 ///////////////
        *(static_cast<void**>(aligned_mem) - 1) = allocated_mem;
        ////////////// #3 ///////////////
        return aligned_mem;
#endif
    }

public:
    CompMatrix()=default;
    ~CompMatrix(){
        free(weight);
        free(rowIdx);
        free(colIdx);
    }

    void read(FILE* fd, int elSize, int _nRows, int _nCols){
        int nWeights=0;
        int nIndex=0;
        nRows = _nRows;
        nCols = _nCols;


        fread(&nWeights, sizeof(int), 1, fd);
        std::vector<float> weight(nWeights);
        fread(weight.data(), elSize, nWeights, fd);

        fread(&nIndex, sizeof(int), 1, fd);
        std::vector<uint8_t> index(nIndex);
        fread(index.data(), sizeof(uint8_t), nIndex, fd);
        prepData( weight, index );
    }

    Vectorf operator*( const Vectorf& x);
};

class BaseLayer {
public:
    BaseLayer() = default;
    virtual BaseLayer* loadNext( FILE* fd ) {assert(0); return nullptr;};
    virtual Matrixf apply( const Matrixf& x){assert(0); return Matrixf();};
    virtual Vectorf apply( const Vectorf& x){assert(0); return Vectorf();};
    virtual Vectorf apply( const Vectorf& x, const Vectorf& h){assert(0); return Vectorf();};
    virtual std::vector<int> shape(void) const { return std::vector<int>(); }

};

//TODO: This should be turned into a proper class factory pattern
class TorchLayer : public BaseLayer {
    struct  Header{
        //int size; //size of data blob, not including this header
        enum class LayerType : int { Conv1d=1, Conv2d=2, BatchNorm1d=3, Linear=4, GRU=5, Stretch2d=6 } layerType;
        char name[64]; //layer name for debugging
    };

    BaseLayer* impl;

public:
    BaseLayer* loadNext( FILE* fd );

    template< typename T> T operator()( const T& x){ return impl->apply( x ); }
    template< typename T, typename T2> T operator()( const T& x, const T2& h ){ return impl->apply( x, h );}
    virtual std::vector<int> shape( void ) const override { return impl->shape(); }

    virtual Matrixf apply( const Matrixf& x) override { return impl->apply(x); };
    virtual Vectorf apply( const Vectorf& x) override { return impl->apply(x); };
    virtual Vectorf apply( const Vectorf& x, const Vectorf& h) override { return impl->apply(x); };

    virtual ~TorchLayer(){
        delete impl;
        impl=nullptr;
    }
};

class Conv1dLayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int useBias;
        int inChannels;
        int outChannels;
        int kernelSize;
    };

    std::vector<Matrixf> weight;
    Vectorf bias;

    bool hasBias;
    int inChannels;
    int outChannels;
    int nKernel;
public:
    Conv1dLayer() = default;
    //call TorchLayer loadNext, not derived loadNext
    Conv1dLayer* loadNext( FILE* fd );
    Matrixf apply( const Matrixf& x ) override;
    virtual std::vector<int> shape( void ) const override { return std::vector<int>({inChannels, outChannels, nKernel}); }
};

class Conv2dLayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int nKernel;  //kernel size. special case of conv2d used in WaveRNN
    };

    Vectorf weight;
    int nKernel;

public:
    Conv2dLayer() = default;
    //call TorchLayer loadNext, not derived loadNext
    Conv2dLayer* loadNext( FILE* fd );
    Matrixf apply( const Matrixf& x ) override;
    virtual std::vector<int> shape(void) const override { return std::vector<int>({nKernel}); }
};

class BatchNorm1dLayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int inChannels;
        float eps;
    };

    Vectorf weight;
    Vectorf bias;
    Vectorf running_mean;
    Vectorf running_var;
    float eps;
    int nChannels;

public:
    //call TorchLayer loadNext, not derived loadNext
    BatchNorm1dLayer* loadNext( FILE* fd );

    Matrixf apply(const Matrixf &x ) override;
    virtual std::vector<int> shape(void) const override { return std::vector<int>({nChannels}); }
};


class LinearLayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int nRows;
        int nCols;
    };

    CompMatrix mat;
    Vectorf bias;
    int nRows;
    int nCols;


public:
    LinearLayer() = default;
    //call TorchLayer loadNext, not derived loadNext
    LinearLayer* loadNext( FILE* fd );
    Vectorf apply( const Vectorf& x ) override;
    virtual std::vector<int> shape(void) const override { return std::vector<int>({nRows, nCols}); }
};


class GRULayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int nHidden;
        int nInput;
    };

    CompMatrix W_ir,W_iz,W_in;
    CompMatrix W_hr,W_hz,W_hn;
    Vectorf b_ir,b_iz,b_in;
    Vectorf b_hr,b_hz,b_hn;
    int nRows;
    int nCols;

public:
    GRULayer() = default;
    //call TorchLayer loadNext, not derived loadNext
    GRULayer* loadNext( FILE* fd );
    Vectorf apply( const Vectorf& x, const Vectorf& hx ) override;
    virtual std::vector<int> shape(void) const override { return std::vector<int>({nRows, nCols}); }
};

class Stretch2dLayer : public TorchLayer{
    struct  Header{
        int x_scale;
        int y_scale;
    };

    int x_scale;
    int y_scale;

public:
    Stretch2dLayer() = default;
    //call TorchLayer loadNext, not derived loadNext
    Stretch2dLayer* loadNext( FILE* fd );
    Matrixf apply(const Matrixf &x ) override;
    virtual std::vector<int> shape(void) const override { return std::vector<int>({0}); }
};


#endif // WAVERNN_H
