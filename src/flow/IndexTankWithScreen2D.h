//
// Created by geoolekom on 18.04.19.
//

#include "interfaces/base.h"

#ifndef CALC_INDEXTANKWITHSCREEN2D_H
#define CALC_INDEXTANKWITHSCREEN2D_H


class IndexTankWithScreen2D {
private:
    int wallLeftX, wallRightX, wallY;  // Правая стенка ящика
    int ceilingY;  // Верхняя стенка
    int screenLeftX, screenRightX, screenY;  // Экран
    int endX;  // Правая граница области счета
public:

    IndexTankWithScreen2D(int wallLeftX, int wallRightX, int wallY, int ceilingY, int screenLeftX,
                     int screenRightX, int screenY, int endX) :
            wallLeftX (wallLeftX), wallRightX (wallRightX), wallY (wallY),
            ceilingY (ceilingY), screenLeftX (screenLeftX), screenRightX (screenRightX), screenY (screenY),
            endX (endX) {};

    bool isDiffuseReflection(const intVector& x, const doubleVector& v) {
        bool value = false;
        if ((x[0] == 0) ||
            (x[0] == wallLeftX && x[1] >= wallY) ||
            (x[0] == wallRightX && x[1] >= wallY) ||
            (x[0] == screenLeftX && x[1] >= screenY) ||
            (x[0] == screenRightX && x[1] >= screenY)) {
            // летящие вправо имеют рассеянное распределение
            value = value || v[0] > 0;
        }
        if ((x[0] == wallLeftX - 1 && x[1] >= wallY) ||
            (x[0] == wallRightX - 1 && x[1] >= wallY) ||
            (x[0] == screenLeftX - 1 && x[1] >= screenY) ||
            (x[0] == screenRightX - 1 && x[1] >= screenY)) {
            // влево
            value = value || v[0] < 0;
        }
        if (((x[0] < wallRightX || (x[0] >= screenLeftX && x[0] < screenRightX)) && x[1] == ceilingY - 1) ||
            ((x[0] >= wallLeftX && x[0] < wallRightX) && x[1] == wallY - 1) ||
            ((x[0] >= screenLeftX && x[0] < screenRightX) && x[1] == screenY - 1)) {
            // вниз
            value = value || v[1] < 0;
        }
        if (((x[0] >= screenLeftX && x[0] < screenRightX) && x[1] == screenY) ||
            ((x[0] >= wallLeftX && x[0] < wallRightX) && x[1] == wallY)) {
            // вверх
            value = value || v[1] > 0;
        }
        return value;
    }

    bool isMirrorReflection(const intVector& x, const doubleVector& v) {
        return x[1] == 0 && v[1] > 0;
    }

    bool isBorderReached(const intVector& x, const doubleVector& v) {
        return (((x[0] >= wallRightX && x[0] < screenLeftX) || x[0] >= screenRightX) && x[1] == ceilingY - 1 && v[1] < 0) ||
                (x[0] == endX - 1 && v[0] < 0);
    }

    bool isInTank(const intVector& x) {
        return x[0] < wallLeftX;
    }
};


#endif //CALC_INDEXTANKWITHSCREEN2D_H
