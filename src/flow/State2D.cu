#ifndef CALC_STATE2D_H
#define CALC_STATE2D_H

#include <vector>
#include <cuda_runtime_api.h>
#include "interfaces/base2D.h"


class State2D {
public:
    typedef std::vector<intVector> iterable;
private:
    double* data;
    iterable spaceIterable, velocityIterable;
public:
    int nvx, nvy, nx, ny;
    int xIndexMax, yIndexMax, vxIndexMin, vxIndexMax, vyIndexMin, vyIndexMax;

    __host__ __device__ inline double getValue(int xIndex, int yIndex, int vxIndex, int vyIndex) {
        if (yIndex == -1) {
            return 2 * data[this->index(xIndex, 0, vxIndex, vyIndex)] - data[this->index(xIndex, 1, vxIndex, vyIndex)];
        } else if (yIndex == yIndexMax) {
            return 2 * data[this->index(xIndex, yIndexMax - 1, vxIndex, vyIndex)] - data[this->index(xIndex, yIndexMax - 2, vxIndex, vyIndex)];
        }
        return data[this->index(xIndex, yIndex, vxIndex, vyIndex)];
    };

    __host__ __device__ inline double getValue(const intVector& xIndex, const intVector& vIndex) {
        return this->getValue(xIndex[0], xIndex[1], vIndex[0], vIndex[1]);
    }

    __host__ __device__ void setValue(int xIndex, int yIndex, int vxIndex, int vyIndex, double value) {
        data[this->index(xIndex, yIndex, vxIndex, vyIndex)] = value;
    };

    __host__ __device__ void setValue(const intVector& xIndex, const intVector& vIndex, double value) {
        data[this->index(xIndex[0], xIndex[1], vIndex[0], vIndex[1])] = value;
    };

    State2D(int xIndexMax, int yIndexMax, int vxIndexMin, int vxIndexMax, int vyIndexMin, int vyIndexMax) :
        xIndexMax(xIndexMax), yIndexMax(yIndexMax),
        vxIndexMin(vxIndexMin), vxIndexMax(vxIndexMax),
        vyIndexMin(vyIndexMin), vyIndexMax(vyIndexMax) {
        this->nx = xIndexMax;
        this->ny = yIndexMax;
        this->nvx = vxIndexMax - vxIndexMin;
        this->nvy = vyIndexMax - vyIndexMin;

        for (int yIndex = 0; yIndex < this->yIndexMax; yIndex ++) {
            for (int xIndex = 0; xIndex < this->xIndexMax; xIndex++) {
                spaceIterable.push_back({xIndex, yIndex});
            }
        }

        for (int vyIndex = this->vyIndexMin; vyIndex < this->vyIndexMax; vyIndex++) {
            for (int vxIndex = this->vxIndexMin; vxIndex < this->vxIndexMax; vxIndex++) {
                velocityIterable.push_back({vxIndex, vyIndex});
            }
        }
    }

    void allocate() {
        this->data = new double[nx * ny * nvx * nvy]();
    }

    void free() {
        delete[] this->data;
    }

    State2D(const State2D& state) : State2D(state.xIndexMax, state.yIndexMax,
            state.vxIndexMin, state.vxIndexMax, state.vyIndexMin, state.vyIndexMax) {}

    ~State2D() {};

    __host__ __device__ inline int index(int xIndex, int yIndex, int vxIndex, int vyIndex) {
        return (vxIndex + nvx) % nvx
            + ((vyIndex + nvy) % nvy) * nvx
            + ((xIndex + nx) % nx) * nvx * nvy
            + ((yIndex + ny) % ny) * nvx * nvy * nx;
    }

    __device__ inline double* velocitySlice(int xIndex, int yIndex) {
        return data + nvx * nvy * (yIndex + xIndex * ny);
    }

    const iterable& getSpaceIterable() {
        return this->spaceIterable;
    }

    const iterable& getVelocityIterable() {
        return this->velocityIterable;
    }

    void setData(double* data) {
        cudaMemcpy(&(this->data), &data, sizeof(double*), cudaMemcpyHostToDevice);
    }

    double* getData() { return this->data; }

    int getSize() { return nx * ny * nvx * nvy; }

};


#endif //CALC_STATE2D_H
