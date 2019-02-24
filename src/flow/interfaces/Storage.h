//
// Created by geoolekom on 24.02.19.
//

#ifndef CALC_STORAGE_H
#define CALC_STORAGE_H

#include <cstddef>
#include "Space.h"
#include "Grid.h"

template <std::size_t dimension>
class Storage {
public:
    typedef Space<dimension> spaceType;
    typedef typename spaceType::vectorType vectorType;
    typedef Grid<dimension> gridType;
private:
    spaceType* state;
    gridType* grid;
public:
    Storage() = default;
    Storage(spaceType* state, gridType* grid) : state(state), grid(grid) {};

    void exportDensity(std::ostream* stream) {
        int vx = 0;
        for (const vectorType& v : state->spaceIterable()) {
            if (vx != v[0]) {
                vx = v[0];
                (*stream) << std::endl;
            }
            (*stream) << grid->getCoordinates(v)
                      << this->getDensity(v) << "\n";

        }
    }

    double getDensity(const vectorType& xIndex) {
        double value = 0;
        for (const vectorType& vIndex : state->velocityIterable()) {
            value += state->getValue(xIndex, vIndex);
        }
        return value;
    }
};


#endif //CALC_STORAGE_H
