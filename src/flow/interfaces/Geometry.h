//
// Created by geoolekom on 21.02.19.
//

#ifndef CALC_GEOMETRY_H
#define CALC_GEOMETRY_H

#include "base.h"


class Geometry2D {
public:

    Geometry2D() = default;

    virtual bool isInTank(const intVector& xIndex) = 0;

    virtual bool isDiffuseReflection(const intVector& xIndex, const doubleVector& v) = 0;

    virtual bool isMirrorReflection(const intVector& xIndex, const doubleVector& v) = 0;

    virtual bool isBorderReached(const intVector& xIndex, const doubleVector& v) = 0;
};


#endif //CALC_GEOMETRY_H
