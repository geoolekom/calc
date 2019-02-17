//
// Created by geoolekom on 23.12.18.
//

#ifndef CALC_STORAGE2D_H
#define CALC_STORAGE2D_H


#include <ostream>
#include "State2D.h"
#include "Grid2D.h"
#include "../../libs/ci/ci.hpp"

class Storage2D {
private:
    State2D* state;
    Grid2D* grid;
public:
    Storage2D() = default;
    Storage2D(State2D* state, Grid2D* grid) : state(state), grid(grid) {};

    void exportDensity(std::ostream* stream) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex ++) {
            for (int yIndex = 0; yIndex < state->yIndexMax; yIndex ++) {
                (*stream)
                        << grid->getX(xIndex) << "\t"
                        << grid->getY(yIndex) << "\t"
                        << this->getDensity(xIndex, yIndex) << "\n";
            }
            (*stream) << std::endl;
        }
    }

    double getDensity(int xIndex, int yIndex) {
        double value = 0, vx, vy;
        for (int vyIndex = state->vyIndexMin; vyIndex < state->vyIndexMax; vyIndex ++) {
            for (int vxIndex = state->vxIndexMin; vxIndex < state->vxIndexMax; vxIndex ++) {
                value += state->getValue(xIndex, yIndex, vxIndex, vyIndex);
            }
        }
        return value;
    }

    void exportTemperature(std::ostream* stream) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex ++) {
            for (int yIndex = 0; yIndex < state->yIndexMax; yIndex ++) {
                (*stream)
                        << grid->getX(xIndex) << "\t"
                        << grid->getY(yIndex) << "\t"
                        << this->getTemperature(xIndex, yIndex) << "\n";
            }
            (*stream) << std::endl;
        }
    }

    double getTemperature(int xIndex, int yIndex) {
        double nom = 0, denom = 0, vx, vy;
        for (int vyIndex = state->vyIndexMin; vyIndex < state->vyIndexMax; vyIndex ++) {
            for (int vxIndex = state->vxIndexMin; vxIndex < state->vxIndexMax; vxIndex ++) {
                vx = grid->getVx(vxIndex);
                vy = grid->getVy(vyIndex);
                denom += state->getValue(xIndex, yIndex, vxIndex, vyIndex);
                nom += (vx * vx + vy * vy) * state->getValue(xIndex, yIndex, vxIndex, vyIndex);
            }
        }
        return nom / denom;
    }
};


#endif //CALC_STORAGE2D_H
