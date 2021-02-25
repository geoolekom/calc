//
// Created by geoolekom on 26.02.2021.
//

#ifndef CALC_PARTIALCELLTANK_CUH
#define CALC_PARTIALCELLTANK_CUH

#include "../interfaces/base3D.cu"

class PartialCellTank {
  private:
    int holeCenterY, holeCenterZ, holeRadius; // Центр и радиус отверстия
    int wallLeftX, wallRightX;                // Правая стенка ящика
    floatType *intersections;

  public:
    PartialCellTank(int holeCenterY, int holeCenterZ, int holeRadius, int wallLeftX, int wallRightX);
    ~PartialCellTank();
    __device__ double getFreeFlowArea(const intVector &x) const;
    __device__ double getDiffusionArea(const intVector &x) const;
};

#endif // CALC_PARTIALCELLTANK_CUH
