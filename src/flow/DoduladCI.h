//
// Created by geoolekom on 20.02.19.
//

#ifndef CALC_DODULADCI_H
#define CALC_DODULADCI_H


#include "../../libs/ci/ci.hpp"
#include "IndexMap.h"
#include "State2D.h"

#define KOROBOV_GRID_PARAMETER 50000

class DoduladCI {
private:
    double particleDiameter = 1.0, particleMass = 1.0, vStep, tStep;
    int vGridRadius;
    ci::HSPotential potential;
    ci::Particle particle = {particleDiameter};
    IndexMap* speedIndexMap;
    State2D* state;
public:
    DoduladCI(double tStep, double vStep, State2D* state) : tStep(tStep), vStep(vStep), state(state) {
        ci::init(&potential, ci::Z_SYMM);
        vGridRadius = state->vxIndexMax - 1;
        speedIndexMap = new IndexMap(state->vxIndexMax - state->vxIndexMin, state->vyIndexMax - state->vyIndexMin);
    }

    ~DoduladCI() {
        delete speedIndexMap;
    }

    void timeGenerator() {
        ci::gen(tStep, KOROBOV_GRID_PARAMETER, vGridRadius, vGridRadius,
                *speedIndexMap, *speedIndexMap, vStep, particleMass, particleMass, particle, particle);
    };
    void calculateIntegral(int xIndex, int yIndex) {
        auto slice = state->velocitySlice(xIndex, yIndex);
        ci::iter(slice, slice);
    };
};


#endif //CALC_DODULADCI_H
