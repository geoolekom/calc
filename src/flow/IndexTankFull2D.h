//
// Created by geoolekom on 22.04.19.
//
#include "interfaces/base2D.h"

#ifndef CALC_INDEXTANKFULL2D_H
#define CALC_INDEXTANKFULL2D_H


class IndexTankFull2D {
private:
    int wallLeftX, wallRightX, wallBottomY, wallTopY;  // Правая стенка ящика
    int ceilingY;  // Верхняя стенка
    int endX;  // Правая граница области счета
public:

    IndexTankFull2D(int wallLeftX, int wallRightX, int wallBottomY, int wallTopY, int ceilingY, int endX) :
            wallLeftX (wallLeftX), wallRightX (wallRightX), wallBottomY (wallBottomY), wallTopY(wallTopY),
            ceilingY (ceilingY), endX (endX) {};

    bool isDiffuseReflection(const intVector& x, const doubleVector& v) {
        bool value = false;
        if ((x[0] == 0) ||
            (x[0] == wallLeftX && (x[1] >= wallTopY || x[1] < wallBottomY)) ||
            (x[0] == wallRightX && (x[1] >= wallTopY || x[1] < wallBottomY))) {
            // летящие вправо имеют рассеянное распределение
            value = value || v[0] > 0;
        }
        if ((x[0] == wallLeftX - 1 && (x[1] >= wallTopY || x[1] < wallBottomY)) ||
            (x[0] == wallRightX - 1 && (x[1] >= wallTopY || x[1] < wallBottomY))) {
            // влево
            value = value || v[0] < 0;
        }
        if ((x[0] < wallRightX && x[1] == ceilingY - 1) ||
            ((x[0] >= wallLeftX && x[0] < wallRightX) && (x[1] == wallTopY - 1 || x[1] == wallBottomY - 1))) {
            // вниз
            value = value || v[1] < 0;
        }
        if (((x[0] >= wallLeftX && x[0] < wallRightX) && (x[1] == wallTopY || x[1] == wallBottomY)) ||
            (x[1] == 0 && x[0] < wallRightX)) {
            // вверх
            value = value || v[1] > 0;
        }
        return value;
    }

    bool isMirrorReflection(const intVector& x, const doubleVector& v) {
        return false;
    }

    bool isBorderReached(const intVector& x, const doubleVector& v) {
        return (x[0] >= wallRightX && x[1] == ceilingY - 1 && v[1] < 0) ||
               (x[0] == endX - 1 && v[0] < 0) ||
               (x[1] == 0 && x[0] >= wallRightX && v[1] > 0);
    }

    bool isInTank(const intVector& x) {
        return x[0] < wallLeftX;
    }
};


#endif //CALC_INDEXTANKFULL2D_H
