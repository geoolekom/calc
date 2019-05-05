//
// Created by geoolekom on 27.04.19.
//

#ifndef CALC_GRID3D_H
#define CALC_GRID3D_H

#include "interfaces/base3D.h"
#include <host_defines.h>

class Grid3D {
private:
    const double vLimitSqr;
    const double vShift = 0.5;

    __host__ __device__ double toValue(int index, double step, double shift) const;
    __host__ __device__ int toIndex(double value, double step, double shift) const;
public:
    const double vxStep, vyStep, vzStep;
    const double xStep, yStep, zStep;

    Grid3D(const doubleVector& xStep, const doubleVector& vStep, double vLimit);
    ~Grid3D() = default;

    __host__ __device__ doubleVector getX(const intVector& index) const;
    __host__ __device__ doubleVector getV(const intVector& index) const;

    __host__ __device__ double getXx(int index) const;
    __host__ __device__ double getXy(int index) const;
    __host__ __device__ double getXz(int index) const;

    __host__ __device__ double getVx(int index) const;
    __host__ __device__ double getVy(int index) const;
    __host__ __device__ double getVz(int index) const;

    __host__ __device__ bool inBounds(const intVector& xIndex) const;
    __host__ __device__ bool inBounds(const doubleVector& xIndex) const;

    __host__ __device__ intVector getVIndex(const doubleVector& v);
};


#endif //CALC_GRID3D_H
