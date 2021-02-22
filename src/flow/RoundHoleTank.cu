//
// Created by geoolekom on 10.02.2021.
//

#include "RoundHoleTank.cuh"

RoundHoleTank::RoundHoleTank(int holeCenterY, int holeCenterZ, int holeRadius, int wallLeftX, int wallRightX,
                             int ceilingY, int ceilingZ, int endX)
    : holeCenterY(holeCenterY), holeCenterZ(holeCenterZ), holeRadius(holeRadius), wallLeftX(wallLeftX),
      wallRightX(wallRightX), ceilingY(ceilingY), ceilingZ(ceilingZ), endX(endX){};

__device__ bool RoundHoleTank::inHoleNeighbourhood(const intVector &x) const {
    const int distanceSqr = (x[1] - holeCenterY) * (x[1] - holeCenterY) + (x[2] - holeCenterZ) * (x[2] - holeCenterZ);
    return distanceSqr < holeRadius * holeRadius;
}

__device__ bool RoundHoleTank::isDiffuseReflection(const intVector &x, const doubleVector &v) const {
    // Тут нарочно не рассматриваем отражение от боковой грани отверстия
    bool value = false;
    const bool isInHoleNeighbourhood = this->inHoleNeighbourhood(x);
    if ((x[0] == 0) || (x[0] == wallLeftX && !isInHoleNeighbourhood) ||
        (x[0] == wallRightX && !isInHoleNeighbourhood)) {
        // летящие вправо имеют рассеянное распределение
        value = v[0] > 0;
    };
    if ((x[0] == wallLeftX - 1 && !isInHoleNeighbourhood) || (x[0] == wallRightX - 1 && !isInHoleNeighbourhood)) {
        // влево
        value = value || v[0] < 0;
    }
    if (x[1] == ceilingY - 1 && x[0] < wallRightX) {
        // вниз
        value = value || v[1] < 0;
    }
    if (x[1] == 0 && x[0] < wallRightX) {
        // вверх
        value = value || v[1] > 0;
    }
    if (x[2] == 0 && x[0] < wallRightX) {
        // ближняя стенка по Z
        value = value || v[2] > 0;
    }
    if (x[2] == ceilingZ && x[0] < wallRightX) {
        // дальняя стенка по Z
        value = value || v[2] < 0;
    }
    return value;
}

__device__ bool RoundHoleTank::isMirrorReflection(const intVector &x, const doubleVector &v) const { return false; }

__device__ bool RoundHoleTank::isBorderReached(const intVector &x, const doubleVector &v) const {
    return (x[0] >= wallRightX && x[1] == ceilingY - 1 && v[1] < 0) || (x[0] >= wallRightX && x[1] == 0 && v[1] > 0) ||
           (x[0] >= wallRightX && x[2] == ceilingZ - 1 && v[2] < 0) || (x[0] >= wallRightX && x[2] == 0 && v[2] > 0) ||
           (x[0] == endX - 1 && v[0] < 0);
}

__device__ bool RoundHoleTank::isFreeFlow(const intVector &x, const doubleVector &v) const {
    return !(this->isDiffuseReflection(x, v) || this->isMirrorReflection(x, v) || this->isBorderReached(x, v));
}

__host__ __device__ bool RoundHoleTank::isInTank(const intVector &x) const { return x[0] < wallLeftX; }