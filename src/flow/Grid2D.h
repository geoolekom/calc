//
// Created by geoolekom on 21.12.18.
//

#ifndef CALC_GRID2D_H
#define CALC_GRID2D_H


class Grid2D {
public:
    double xStep, yStep, vxStep, vyStep;

    Grid2D(double xStep, double yStep, double vxStep, double vyStep) :
            xStep(xStep), yStep(yStep), vxStep(vxStep), vyStep(vyStep) {};

    ~Grid2D() = default;

    inline double getX(int xIndex) {
        return xIndex * xStep;
    }

    inline double getY(int yIndex) {
        return yIndex * yStep;
    }

    inline double getVx(int vxIndex) {
        return vxIndex * vxStep;
    }

    inline double getVy(int vyIndex) {
        return vyIndex * vyStep;
    }

    inline int getXIndex(double x) {
        return int (x / xStep);
    }

    inline int getYIndex(double y) {
        return int (y / yStep);
    }
};


#endif //CALC_GRID2D_H
