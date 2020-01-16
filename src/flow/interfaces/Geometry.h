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

    virtual __host__ __device__ bool isInTank(const intVector& xIndex) { return false; };

    virtual __device__ bool isDiffuseReflection(const intVector& xIndex, const doubleVector& v) { printf("VIRTUAL\n"); return false; };

    virtual __device__ bool isMirrorReflection(const intVector& xIndex, const doubleVector& v) { return false; };

    virtual __device__ bool isBorderReached(const intVector& xIndex, const doubleVector& v) { return true; };
};


#endif //CALC_GEOMETRY_H
