//
// Created by geoolekom on 21.02.19.
//

#ifndef CALC_GEOMETRY_H
#define CALC_GEOMETRY_H

#include <array>


template <std::size_t dimension>
class Geometry {
public:
    typedef std::array<int, dimension> Vector;
    typedef std::array<std::array<int, dimension>, dimension> Matrix;

    Geometry() = default;

    virtual bool isDiffuseReflection(const Vector& xIndex, const Vector& vIndex) = 0;

    virtual bool isMirrorReflection(const Vector& xIndex, const Vector& vIndex) = 0;

    virtual bool isBorderReached(const Vector& xIndex) = 0;

    virtual Matrix getMirrorNormal(const Vector& xIndex) = 0;

    virtual Vector getDiffusionMask(const Vector& xIndex) = 0;
};


#endif //CALC_GEOMETRY_H
