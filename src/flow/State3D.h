#ifndef CALC_STATE3D_CU
#define CALC_STATE3D_CU

#include "interfaces/base3D.h"
#include <host_defines.h>
#include <vector>

class State3D {
private:
    typedef std::vector<intVector> iterable;

    double* data;
    iterable spaceIterable, velocityIterable;
public:
    const int xIndexMax, yIndexMax, zIndexMax;
    const int vxIndexMin, vyIndexMin, vzIndexMin;
    const int vxIndexMax, vyIndexMax, vzIndexMax;
    const int nx, ny, nz, nvx, nvy, nvz;

    State3D(intVector xLimits, intVector vBottomLimits, intVector vTopLimits);
    State3D(const State3D& state);
    ~State3D() = default;

    int getSize() const;

    double* getData() const;
    void setData(double* data);
    void cudaSetData(double* data);

    void allocate();
    void cudaAllocate();

    void release();
    void cudaRelease();

    __host__ __device__ int index(const intVector &x, const intVector &v) const;
    __host__ __device__ double getValue(const intVector &x, const intVector &v) const;
    __host__ __device__ void setValue(const intVector &x, const intVector &v, double value);
    __device__ double* getVelocitySlice(const intVector &x);

    const iterable& getSpaceIterable();
    const iterable& getVelocityIterable();

};

#endif // CALC_STATE3D_CU
