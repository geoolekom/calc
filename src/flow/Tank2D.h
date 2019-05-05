//
// Created by geoolekom on 21.12.18.
//

#ifndef CALC_TANK2D_H
#define CALC_TANK2D_H
#define EPS 1e-3


#include <cmath>
#include <iostream>
#include "interfaces/base2D.h"

class Tank2D {
private:
    double wallLeftX, wallRightX, wallY;  // Правая стенка ящика
    double ceilingY;  // Верхняя стенка
    double endX;  // Правая граница области счета
public:

    Tank2D(double wallLeftX, double wallRightX, double wallY, double ceilingY, double endX) :
            wallLeftX (wallLeftX), wallRightX (wallRightX), wallY (wallY),
            ceilingY (ceilingY), endX (endX) {};

    bool isDiffuseReflection(const doubleVector& x, const doubleVector& v) {
        bool value = false;
        if ((std::abs(x[0]) < EPS) ||
            (std::abs(x[0] - wallRightX) < EPS && (x[1] > wallY || std::abs(x[1] - wallY) < EPS))) {
            // летящие вправо имеют рассеянное распределение
            value = value || v[0] > 0;
        }
        if ((std::abs(x[0] - wallLeftX) < EPS && (x[1] > wallY || std::abs(x[1] - wallY) < EPS))) {
            // влево
            value = value || v[0] < 0;
        }
        if ((std::abs(x[1] - ceilingY) < EPS && (x[0] < wallLeftX || std::abs(x[0] - wallLeftX) < EPS)) ||
            (std::abs(x[1] - wallY) < EPS && x[0] >= wallLeftX && x[0] <= wallRightX)) {
            // вниз
            value = value || v[1] < 0;
        }
//        if ((x[0] > wallRightX || std::abs(x[0] - wallRightX) < EPS) && std::abs(x[1] - ceilingY) < EPS) {
//            value = value || v[1] < 0;
//        }
//        if (std::abs(x[0] - endX) < EPS) {
//            value = value || v[0] < 0;
//        }
        return value;
    }

    bool isMirrorReflection(const doubleVector& x, const doubleVector& v) {
        return std::abs(x[1]) < EPS && v[1] > 0;
    }

    bool isBorderReached(const doubleVector& x, const doubleVector& v) {
        return ((x[0] > wallRightX || std::abs(x[0] - wallRightX) < EPS) && std::abs(x[1] - ceilingY) < EPS && v[1] < 0) ||
                (std::abs(x[0] - endX) < EPS && v[0] < 0);
    }

    bool isInTank(const doubleVector& x) {
        return x[0] < wallLeftX || std::abs(x[0] - wallLeftX) < EPS;
    }
};


#endif //CALC_TANK2D_H
