//
// Created by geoolekom on 10.02.2021.
//

#include "RoundHoleTank.cuh"

RoundHoleTank::RoundHoleTank(int holeCenterY, int holeCenterZ, int holeRadius, int wallLeftX, int wallRightX,
                             int ceilingY, int ceilingZ, int endX)
    : holeCenterY(holeCenterY), holeCenterZ(holeCenterZ), holeRadius(holeRadius), wallLeftX(wallLeftX),
      wallRightX(wallRightX), ceilingY(ceilingY), ceilingZ(ceilingZ), endX(endX){};

__device__ bool RoundHoleTank::inHoleNeighbourhood(const intVector &x) const {
    const double shift = 0.5;
    const double distanceSqr = (x[1] + shift - holeCenterY) * (x[1] + shift - holeCenterY) +
                               (x[2] + shift - holeCenterZ) * (x[2] + shift - holeCenterZ);
    return distanceSqr < holeRadius * holeRadius;
}

__device__ bool RoundHoleTank::isDiffuseReflection(const intVector &x, const doubleVector &v) const {
    bool value = false;
    const bool isInHoleNeighbourhood = this->inHoleNeighbourhood(x);
    const intVector shiftY = {0, 1, 0};
    const intVector shiftZ = {0, 0, 1};
    if (wallLeftX - 1 < x[0] && x[0] < wallRightX && isInHoleNeighbourhood) {
        // На краю отверстия. Y - по вертикали, Z - по горизонтали.
        const bool wallByLeft = !this->inHoleNeighbourhood(x - shiftZ) && v[2] > 0;
        const bool wallByRight = !this->inHoleNeighbourhood(x + shiftZ) && v[2] < 0;
        const bool wallByBottom = !this->inHoleNeighbourhood(x - shiftY) && v[1] > 0;
        const bool wallByTop = !this->inHoleNeighbourhood(x + shiftY) && v[1] < 0;
        value = wallByLeft || wallByRight || wallByBottom || wallByTop;
    }
    if (wallLeftX - 1 < x[0] && x[0] < wallRightX && !isInHoleNeighbourhood) {
        // Внутри стенки
        const bool wallByLeft = this->inHoleNeighbourhood(x + shiftZ) && v[2] < 0;
        const bool wallByRight = this->inHoleNeighbourhood(x - shiftZ) && v[2] > 0;
        const bool wallByBottom = this->inHoleNeighbourhood(x + shiftY) && v[1] < 0;
        const bool wallByTop = this->inHoleNeighbourhood(x - shiftY) && v[1] > 0;
        value = value || wallByLeft || wallByRight || wallByBottom || wallByTop;
    }
    if ((x[0] == 0) || (x[0] == wallLeftX && !isInHoleNeighbourhood) ||
        (x[0] == wallRightX && !isInHoleNeighbourhood)) {
        // летящие вправо имеют рассеянное распределение
        value = value || v[0] > 0;
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

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
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

void RoundHoleTank::serializeParams(char *outputString) const {
    char format[120] = "holeCenterY: %d, holeCenterZ %d, holeRadius %d, wallLeftX %d, wallRightX %d, ceilingY %d, "
                       "ceilingZ %d, endX %d";
    sprintf(outputString, format, holeCenterY, holeCenterZ, holeRadius, wallLeftX, wallRightX, ceilingY, ceilingZ,
            endX);
}
