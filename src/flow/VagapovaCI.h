//
// Created by geoolekom on 21.02.19.
//

#ifndef CALC_VAGAPOVACI_H
#define CALC_VAGAPOVACI_H

#include <vector>
#include <iterator>
#include "../../libs/relaxation/Relaxation.h"
#include "State2D.h"
#include "interfaces/CollisionIntegral.h"


class VagapovaCI : public CollisionIntegral<State2D> {
private:
    Solid_spheres_one_gas* ci;
    double particleDiameter = 1.0, particleMass = 1.0, vStep, tStep;
    int vGridRadius;
public:
    VagapovaCI(double tStep, double vStep, State2D* state) : tStep(tStep), vStep(vStep) {
        vGridRadius = state->vxIndexMax - 1;
        ci = new Solid_spheres_one_gas(particleDiameter, particleMass, vStep, vGridRadius, tStep);
    }

    ~VagapovaCI() {
        delete ci;
    }

    void stepForward() override {
        ci->createGrid();
    };

    void calculateIntegral(State2D* state, int xIndex, int yIndex) override {
        auto slice = state->velocitySlice(xIndex, yIndex);
        auto ptr = reinterpret_cast<long double*>(slice);
        auto array = new long double[state->nvx * state->nvx * state->nvy];
        for (int i = 0; i < state->nvx; i++) {
            std::copy(ptr, ptr + state->nvx * state->nvy, array + i * state->nvx * state->nvy);
        }
        ci->calculate(array);
        delete[] array;
    };
};


#endif //CALC_VAGAPOVACI_H
