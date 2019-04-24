//
// Created by geoolekom on 21.12.18.
//

#ifndef CALC_GRID2D_H
#define CALC_GRID2D_H

#include "interfaces/base.h"

class Grid2D {
public:
    double xStep, yStep, vxStep, vyStep;
    double vLimitSqr;
    double shift = 0.5;

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
        return (vxIndex + shift) * vxStep;
    }

    inline double getVy(int vyIndex) {
        return (vyIndex + shift) * vyStep;
    }

    inline int getVyIndex(double vy) {
        if (vy > 0) {
            return int (vy / vyStep);
        } else {
            return int (vy / vyStep - 2 * shift);
        }
    }

    inline bool inBounds(const doubleVector& v) {
        const auto len = v * v;
        return len < vLimitSqr && len > 0;
    }

    doubleVector getV(const intVector& index) {
        return { this->getVx(index[0]), this->getVy(index[1]) };
    }

    doubleVector getX(const intVector& index) {
        return { this->getX(index[0]), this->getY(index[1]) };
    }
};


#endif //CALC_GRID2D_H
