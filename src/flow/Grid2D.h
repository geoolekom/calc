//
// Created by geoolekom on 21.12.18.
//

#ifndef CALC_GRID2D_H
#define CALC_GRID2D_H

#include <array>
#include "interfaces/base.h"

class Grid2D {
public:
    double xStep, yStep, vxStep, vyStep;
    double vLimitSqr;

    Grid2D(double xStep, double yStep, double vxStep, double vyStep, double vLimit) :
            xStep(xStep), yStep(yStep), vxStep(vxStep), vyStep(vyStep), vLimitSqr(vLimit * vLimit) {};

    ~Grid2D() = default;

    inline double getX(int xIndex) {
        return xIndex * xStep;
    }

    inline double getY(int yIndex) {
        return yIndex * yStep;
    }

    inline double getVx(int vxIndex) {
        return (vxIndex + 0.5) * vxStep;
    }

    inline double getVy(int vyIndex) {
        return (vyIndex + 0.5) * vyStep;
    }

    inline int getXIndex(double x) {
        return int (x / xStep);
    }

    inline int getYIndex(double y) {
        return int (y / yStep);
    }

    inline bool inBounds(const doubleVector& v) {
        return v[0] * v[0] + v[1] * v[1] < vLimitSqr;
    }

    doubleVector getV(const intVector& index) {
        return { vxStep * (index[0] + 0.5), vxStep * (index[1] + 0.5) };
    }

    doubleVector getX(const intVector& index) {
        return { xStep * index[0], yStep * index[1] };
    }
};


#endif //CALC_GRID2D_H
