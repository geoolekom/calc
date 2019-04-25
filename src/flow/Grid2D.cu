//
// Created by geoolekom on 21.12.18.
//

#ifndef CALC_GRID2D_H
#define CALC_GRID2D_H

#include <host_defines.h>
#include "interfaces/base.h"

class Grid2D {
public:
    double xStep, yStep, vxStep, vyStep;
    double vLimitSqr;
    double shift = 0.5;

    Grid2D(double xStep, double yStep, double vxStep, double vyStep, double vLimit) :
            xStep(xStep), yStep(yStep), vxStep(vxStep), vyStep(vyStep), vLimitSqr(vLimit * vLimit) {};

    ~Grid2D() = default;

    __host__ __device__ inline double getX(int xIndex) {
        return xIndex * xStep;
    }

    __host__ __device__ inline double getY(int yIndex) {
        return yIndex * yStep;
    }

    __host__ __device__ inline double getVx(int vxIndex) {
        return (vxIndex + shift) * vxStep;
    }

    __host__ __device__ inline double getVy(int vyIndex) {
        return (vyIndex + shift) * vyStep;
    }

    __device__ inline int getVyIndex(double vy) {
        if (vy > 0) {
            return int (vy / vyStep);
        } else {
            return int (vy / vyStep - 2 * shift);
        }
    }

    __host__ __device__ inline bool inBounds(const doubleVector& v) {
        const auto len = v * v;
        return len < vLimitSqr && len > 0;
    }

    __host__ __device__ doubleVector getV(const intVector& index) {
        return { this->getVx(index[0]), this->getVy(index[1]) };
    }

    __host__ __device__ doubleVector getX(const intVector& index) {
        return { this->getX(index[0]), this->getY(index[1]) };
    }
};


#endif //CALC_GRID2D_H
