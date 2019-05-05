//
// Created by geoolekom on 27.04.19.
//

#include "Grid3D.h"

Grid3D::Grid3D(const doubleVector &xStep, const doubleVector &vStep, double vLimit) :
    xStep(xStep[0]), yStep(xStep[1]), zStep(xStep[2]),
    vxStep(vStep[0]), vyStep(vStep[1]), vzStep(vStep[2]),
    vLimitSqr(vLimit * vLimit) {};

__host__ __device__ double Grid3D::toValue(int index, double step, double shift) const { return step * (index + shift); };

__host__ __device__ int Grid3D::toIndex(double value, double step, double shift) const {
    if (value > 0) {
        return int (value / step);
    } else {
        return int (value / step - 2 * shift);
    }
};

__host__ __device__ doubleVector Grid3D::getX(const intVector &index) const {
    return { toValue(index[0], xStep, 0), toValue(index[1], yStep, 0), toValue(index[2], zStep, 0) };
};


__host__ __device__ doubleVector Grid3D::getV(const intVector &index) const {
    return { toValue(index[0], vxStep, 0.5), toValue(index[1], vyStep, 0.5), toValue(index[2], vzStep, 0.5) };
};


__host__ __device__ bool Grid3D::inBounds(const intVector &vIndex) const {
    auto v = this->getV(vIndex);
    return v * v > 0 && v * v < vLimitSqr;
}

__host__ __device__ bool Grid3D::inBounds(const doubleVector &v) const {
    return v * v > 0 && v * v < vLimitSqr;
}

__host__ __device__ intVector Grid3D::getVIndex(const doubleVector &v) {
    return { toIndex(v[0], vxStep, 0.5), toIndex(v[1], vyStep, 0.5), toIndex(v[2], vzStep, 0.5) };
};

__host__ __device__ double Grid3D::getXx(int index) const { return this->toValue(index, xStep, 0); };

__host__ __device__ double Grid3D::getXy(int index) const { return this->toValue(index, yStep, 0); };

__host__ __device__ double Grid3D::getXz(int index) const { return this->toValue(index, zStep, 0); };

__host__ __device__ double Grid3D::getVx(int index) const { return this->toValue(index, vxStep, vShift); };

__host__ __device__ double Grid3D::getVy(int index) const { return this->toValue(index, vyStep, vShift); };

__host__ __device__ double Grid3D::getVz(int index) const { return this->toValue(index, vzStep, vShift); };