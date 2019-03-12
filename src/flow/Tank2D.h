//
// Created by geoolekom on 21.12.18.
//

#ifndef CALC_TANK2D_H
#define CALC_TANK2D_H
#define EPS 1e-3


#include <cmath>
#include <array>
#include <iostream>
#include "interfaces/base.h"

class Tank2D {
private:
    double wallLeftX, wallRightX, wallY;  // Правая стенка ящика
    double ceilingY;  // Верхняя стенка
    double screenLeftX, screenRightX, screenY;  // Экран
    double endX;  // Правая граница области счета
public:

    Tank2D(double wallLeftX, double wallRightX, double wallY, double ceilingY, double screenLeftX,
           double screenRightX, double screenY, double endX) :
            wallLeftX (wallLeftX), wallRightX (wallRightX), wallY (wallY),
            ceilingY (ceilingY), screenLeftX (screenLeftX), screenRightX (screenRightX), screenY (screenY),
            endX (endX) {};

    bool isDiffuseReflection(const doubleVector& x, const doubleVector& v) {
        bool value = false;
        if ((std::abs(x[0]) < EPS) ||
            (std::abs(x[0] - wallRightX) < EPS && (x[1] > wallY || std::abs(x[1] - wallY) < EPS)) ||
            (std::abs(x[0] - screenRightX) < EPS && (x[1] > screenY || std::abs(x[1] - screenY) < EPS))) {
            // летящие вправо имеют рассеянное распределение
            value = value || v[0] > 0;
        }
        if ((std::abs(x[0] - wallLeftX) < EPS && (x[1] > wallY || std::abs(x[1] - wallY) < EPS)) ||
            (std::abs(x[0] - screenLeftX) < EPS && (x[1] > screenY || std::abs(x[1] - screenY) < EPS))) {
            // влево
            value = value || v[0] < 0;
        }
        if (std::abs(x[1] - ceilingY) < EPS && (x[0] < wallLeftX || std::abs(x[0] - wallLeftX) < EPS)) {
            // вниз
            value = value || v[1] < 0;
        }
        return value;
    }

    bool isMirrorReflection(const doubleVector& x, const doubleVector& v) {
        return std::abs(x[1]) < EPS && v[1] > 0;
    }

    bool isBorderReached(const doubleVector& x) {
        return ((x[0] > wallRightX || std::abs(x[0] - wallRightX) < EPS) && std::abs(x[1] - ceilingY) < EPS) ||
            std::abs(x[0] - endX) < EPS;
    }

    bool isInTank(const doubleVector& x) {
        return x[0] < wallLeftX || std::abs(x[0] - wallLeftX) < EPS;
    }
};


#endif //CALC_TANK2D_H
