//
// Created by geoolekom on 21.12.18.
//

#ifndef CALC_TANK2D_H
#define CALC_TANK2D_H


class Tank2D {
public:
    
    int rightWallLeftX, rightWallRightX, rightWallY;  // Правая стенка
    int topWallY;  // Верхняя стенка
    int screenLeftX, screenRightX, screenY;  // Экран
    int tankRightX;  // Правая граница области счета

    Tank2D(int rightWallLeftX, int rightWallRightX, int rightWallY, int topWallY, int screenLeftX,
            int screenRightX, int screenY, int tankRightX) :
            rightWallLeftX (rightWallLeftX), rightWallRightX (rightWallRightX), rightWallY (rightWallY),
            topWallY (topWallY), screenLeftX (screenLeftX), screenRightX (screenRightX), screenY (screenY),
            tankRightX (tankRightX) {};

    bool isDiffuseReflection(int xIndex, int yIndex, int vxIndex, int vyIndex) {
        if (xIndex == 0 ||
            (xIndex == rightWallRightX && yIndex >= rightWallY) ||
            (xIndex == screenRightX && yIndex >= screenY)) {
            // летящие вправо имеют рассеянное распределение
            return vxIndex > 0;
        } else if ((xIndex == rightWallLeftX && yIndex >= rightWallY) ||
                   (xIndex == screenLeftX && yIndex >= screenY)) {
            // влево
            return vxIndex < 0;
        } else if (yIndex == topWallY - 1 && xIndex <= rightWallLeftX) {
            // вниз
            return vyIndex < 0;
        } else {
            return false;
        }
    }

    bool isMirrorReflection(int xIndex, int yIndex, int vxIndex, int vyIndex) {
        return yIndex == 0 && vyIndex > 0;
    }

    bool isBorderReached(int xIndex, int yIndex) {
        return (xIndex > rightWallRightX && yIndex == topWallY - 1) || (xIndex == tankRightX - 1);
    }
};


#endif //CALC_TANK2D_H
