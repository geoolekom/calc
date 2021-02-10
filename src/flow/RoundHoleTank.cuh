//
// Created by geoolekom on 10.02.2021.
//

#include <cuda_runtime_api.h>

#include "interfaces/base3D.cu"

#ifndef CALC_ROUNDHOLETANK_CUH
#define CALC_ROUNDHOLETANK_CUH

class RoundHoleTank {
  private:
    int holeCenterX, holeCenterZ, holeRadius; // Центр и радиус отверстия
    int wallLeftX, wallRightX;                // Правая стенка ящика
    int ceilingY;                             // Верхняя стенка
    int endX;                                 // Правая граница области счета
  public:
    RoundHoleTank(int holeCenterX, int holeCenterZ, int holeRadius, int wallLeftX, int wallRightX, int ceilingY,
                  int endX);
    ~RoundHoleTank() = default;

    __device__ bool isDiffuseReflection(const intVector &x, const doubleVector &v) const;
    __device__ bool isMirrorReflection(const intVector &x, const doubleVector &v) const;
    __device__ bool isBorderReached(const intVector &x, const doubleVector &v) const;
    __device__ bool isFreeFlow(const intVector &x, const doubleVector &v) const;
    __host__ __device__ bool isInTank(const intVector &x) const;

    __device__ bool inHoleNeighbourhood(const intVector &x) const;
};

#endif // CALC_ROUNDHOLETANK_CUH
