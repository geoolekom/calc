#include <cuda_runtime_api.h>
#include "State3D.h"

State3D::State3D(intVector xLimits, intVector vBottomLimits, intVector vTopLimits) :
        xIndexMax(xLimits[0]), yIndexMax(xLimits[1]), zIndexMax(xLimits[2]),
        vxIndexMin(vBottomLimits[0]), vyIndexMin(vBottomLimits[1]), vzIndexMin(vBottomLimits[2]),
        vxIndexMax(vTopLimits[0]), vyIndexMax(vTopLimits[1]), vzIndexMax(vTopLimits[2]),
        nx(xLimits[0]), ny(xLimits[1]), nz(xLimits[2]),
        nvx(vTopLimits[0] - vBottomLimits[0]), nvy(vTopLimits[1] - vBottomLimits[1]), nvz(vTopLimits[2] - vBottomLimits[2]) {

    for (int zIndex = 0; zIndex < this->zIndexMax; zIndex ++) {
        for (int yIndex = 0; yIndex < this->yIndexMax; yIndex++) {
            for (int xIndex = 0; xIndex < this->xIndexMax; xIndex++) {
                spaceIterable.push_back({xIndex, yIndex, zIndex});
            }
        }
    }
    for (int zIndex = vzIndexMin; zIndex < this->vzIndexMax; zIndex ++) {
        for (int yIndex = vyIndexMin; yIndex < this->vyIndexMax; yIndex++) {
            for (int xIndex = vxIndexMin; xIndex < this->vxIndexMax; xIndex++) {
                velocityIterable.push_back({xIndex, yIndex, zIndex});
            }
        }
    }
};


State3D::State3D(const State3D &state) : State3D(intVector({xIndexMax, yIndexMax, zIndexMax}),
        intVector({vxIndexMin, vyIndexMin, vzIndexMin}), intVector({vxIndexMax, vyIndexMax, vzIndexMax})) {};

size_t State3D::getSize() const { return nx * ny * nz * nvx * nvy * nvz; };

__host__ __device__ double State3D::getValue(const intVector &x, const intVector &v) const {
    return this->data[this->index(x, v)];
};

__host__ __device__ void State3D::setValue(const intVector &x, const intVector &v, double value) {
    this->data[this->index(x, v)] = (floatType) value;
};

__host__ __device__ int State3D::index(const intVector &x, const intVector &v) const {
    return ((v[0] + nvx) % nvx) * 1 +
           ((v[1] + nvy) % nvy) * nvx +
           ((v[2] + nvz) % nvz) * nvx * nvy +
           ((x[0] + nx) % nx) * nvx * nvy * nvz +
           ((x[1] + ny) % ny) * nvx * nvy * nvz * nx +
           ((x[2] + nz) % nz) * nvx * nvy * nvz * nx * ny;
};

__device__ floatType* State3D::getVelocitySlice(const intVector &x) {
    return data + nvz * nvy * nvx * (((x[0] + nx) % nx) + ((x[1] + ny) % ny) * nx + ((x[2] + nz) % nz) * nx * ny);
}

floatType* State3D::getData() const { return this->data; };

void State3D::setData(floatType* data) {
    this->data = data;
};

void State3D::cudaSetData(floatType* data) {
    cudaMemcpy(&(this->data), &data, sizeof(floatType*), cudaMemcpyHostToDevice);
};

void State3D::cudaAllocate() {
    floatType* data;
    cudaMallocManaged((void**) &data, sizeof(floatType) * this->getSize(), cudaMemAttachGlobal);
    this->cudaSetData(data);
};

void State3D::cudaRelease(){
    cudaFree(this->data);
};

const State3D::iterable& State3D::getSpaceIterable() { return spaceIterable; };

const State3D::iterable& State3D::getVelocityIterable(){ return velocityIterable; };

void State3D::allocate() {
    this->data = new floatType[this->getSize()]();
}

void State3D::release() {
    delete[] this->data;
}
