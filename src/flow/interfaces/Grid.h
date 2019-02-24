//
// Created by geoolekom on 24.02.19.
//

#ifndef CALC_GRID_H
#define CALC_GRID_H

#include <cstddef>
#include <array>

template <std::size_t dimension>
class Grid {
public:
    typedef std::array<int, dimension> IndexVector;
    typedef std::array<double, dimension> RealVector;

    virtual RealVector getVelocity(const IndexVector& vIndex) = 0;
    virtual RealVector getCoordinates(const IndexVector& xIndex) = 0;

};

#endif //CALC_GRID_H
