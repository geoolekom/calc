//
// Created by geoolekom on 26.02.2021.
//

#include "PartialCellTank.cuh"

PartialCellTank::PartialCellTank(int holeCenterY, int holeCenterZ, int holeRadius, int wallLeftX, int wallRightX)
    : holeCenterY(holeCenterY), holeCenterZ(holeCenterZ), holeRadius(holeRadius), wallLeftX(wallLeftX),
      wallRightX(wallRightX) {
    intersections = new floatType[holeRadius + 1];
    auto shift = 0.5;
    for (int i = 0; i < holeRadius + 1; i++) {
        intersections[i] = std::sqrt(holeRadius * holeRadius - (i - shift) * (i - shift));
    }
};

PartialCellTank::~PartialCellTank() { delete intersections; }

__device__ double PartialCellTank::getFreeFlowArea(const intVector &x) const {
    // Для учета других секторов окружности делаем при необходимости замену Z -> -Z, Y -> -Y
    bool mirroredY = false, mirroredZ = false;
    int newY, newZ;
    if (holeCenterY - holeRadius < x[1] && x[1] < holeCenterY) {
        newY = 2 * holeCenterY - x[1];
        mirroredY = true;
    }
    if (holeCenterZ - holeRadius < x[2] && x[2] < holeCenterZ) {
        newZ = 2 * holeCenterZ - x[2];
        mirroredZ = true;
    }
    if (mirroredY || mirroredZ) {
        // Делаем проверку для повернутых индексов
        return this->getFreeFlowArea({x[0], newY, newZ});
    }

    if (x[1] > holeCenterY + holeRadius || x[2] > holeCenterZ + holeRadius) {
        // Точка за пределами окружности
        return x[0] < wallLeftX || x[0] > wallRightX ? 1. : 0.;
    }
    auto shift = 0.5;
    auto leftY = intersections[x[1]], rightY = intersections[x[1] + 1];
    auto bottomZ = intersections[x[2]], topZ = intersections[x[2] + 1];

    auto hasLeftY = x[1] - shift < leftY && leftY < x[1] + shift;
    auto hasRightY = x[1] - shift < rightY && rightY < x[1] + shift;
    auto hasBottomZ = x[2] - shift < bottomZ && bottomZ < x[2] + shift;
    auto hasTopZ = x[2] - shift < topZ && topZ < x[2] + shift;

    if (hasLeftY && hasRightY) {
        // Y-trapezoid
        return (leftY + rightY - 2. * x[1] + 1) / 2.;
    }

    if (hasLeftY && hasBottomZ) {
        // Triangle
        return (leftY - x[1] + shift) * (bottomZ - x[2] + shift) / 2.;
    }

    if (hasTopZ && hasRightY) {
        // Square - triangle
        return 1. - (rightY - x[1] - shift) * (topZ - x[2] - shift) / 2.;
    }

    if (hasTopZ && hasBottomZ) {
        // Z-trapezoid
        return (topZ + bottomZ - 2. * x[2] + 1) / 2.;
    }

    // Все возможные пересечения с окружностью закончились.
    // Клетка может быть или полностью внутри, или полностью снаружи.
    return x[1] < leftY ? 1. : 0.;
}

__device__ double PartialCellTank::getDiffusionArea(const intVector &x) const { return 1. - this->getFreeFlowArea(x); }
