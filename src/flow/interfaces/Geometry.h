//
// Created by geoolekom on 21.02.19.
//

#ifndef CALC_GEOMETRY_H
#define CALC_GEOMETRY_H

#include "base3D.cu"


class Geometry3D {
public:
    Geometry3D() = default;
    ~Geometry3D() = default;

    __device__ virtual bool isDiffuseReflection(const intVector& xIndex, const doubleVector& v) const = 0;
    __device__ virtual bool isMirrorReflection(const intVector& xIndex, const doubleVector& v) const = 0;
    __device__ virtual bool isBorderReached(const intVector& xIndex, const doubleVector& v) const = 0;
    __device__ virtual bool isFreeFlow(const intVector& x, const doubleVector& v) const = 0;
    __host__ __device__ virtual bool isInTank(const intVector& xIndex) const = 0;
};


#endif //CALC_GEOMETRY_H
