#ifndef CALC_STATE3D_CU
#define CALC_STATE3D_CU

#include "interfaces/base3D.cu"
#include <cuda_runtime_api.h>
#include <vector>

class State3D {
private:
    typedef std::vector<intVector> iterable;

    floatType* data;
    iterable spaceIterable, velocityIterable;
public:
    const int xIndexMax, yIndexMax, zIndexMax;
    const int vxIndexMin, vyIndexMin, vzIndexMin;
    const int vxIndexMax, vyIndexMax, vzIndexMax;
    const int nx, ny, nz, nvx, nvy, nvz;

    State3D(intVector xLimits, intVector vBottomLimits, intVector vTopLimits);
    State3D(const State3D& state);
    ~State3D() = default;

    size_t getSize() const;

    floatType* getData() const;
    void setData(floatType* data);
    void cudaSetData(floatType* data);

    void allocate();
    void cudaAllocate();

    void release();
    void cudaRelease();

    __host__ __device__ int index(const intVector &x, const intVector &v) const;
    __host__ __device__ double getValue(const intVector &x, const intVector &v) const;
    __host__ __device__ void setValue(const intVector &x, const intVector &v, double value);
    __device__ floatType* getVelocitySlice(const intVector &x);

    const iterable& getSpaceIterable();
    const iterable& getVelocityIterable();

};

#endif // CALC_STATE3D_CU
