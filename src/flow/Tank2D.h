//
// Created by geoolekom on 21.12.18.
//

#ifndef CALC_TANK2D_H
#define CALC_TANK2D_H


class Tank2D {
public:
    double xWallStart, xWallEnd, yWallStart;

    Tank2D(double xWallStart, double xWallEnd, double yWallStart) :
            xWallStart(xWallStart), xWallEnd(xWallEnd), yWallStart(yWallStart) {};
};


#endif //CALC_TANK2D_H
