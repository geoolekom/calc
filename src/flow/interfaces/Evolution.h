//
// Created by geoolekom on 20.02.19.
//

#ifndef CALC_EVOLUTION_H
#define CALC_EVOLUTION_H


#include <iostream>
#include "State.h"

class Grid {};

class Geometry {};

template <typename StateType>
class Evolution {
private:
    StateType** state;
    StateType *prev, *curr;
    Geometry *geometry;
    Grid *grid;
public:
    Evolution (StateType** initialState) : state(initialState) {
        long int size = (*initialState)->getSize();
        double* data = new double[size]();
        prev = new StateType(data, size);
    }

    ~Evolution() {
        delete[] prev->getData();
        delete prev;
    }

    void evolve(int numSteps) {
        StateType *temp, *curr;
        for (int i = 0; i < numSteps; i ++) {
            std::cout << i << std::endl;
            temp = curr;
            curr = prev;
            prev = temp;
            this->makeStep(i);
        }
        *state = curr;
    };

    void makeStep(int step) {
        double value, flowFactor;
        bool borderReached;
//    Point mirrorNormal;

        for (SpatialPoint point : *prev) {

            flowFactor = calculateFlowFactor(point);
//        mirrorNormal = geometry->getMirrorNormal(spatialPoint);
//        borderReached = geometry->isBorderReached(spatialPoint);

            for (Velocity velocity : point) {
                std::cout << "velocity " << "\n";
//            if (geometry->isDiffuseReflection(spatialPoint, velocityPoint)) {
//                value = flowFactor * exp(- velocityPoint * velocityPoint / 2);
//            } else if (geometry->isMirrorReflection(spatialPoint, velocityPoint)) {
//                value = prev->getValue(spatialPoint, mirrorNormal * velocityPoint);
//            } else if (borderReached) {
//                value = prev->getValue(spatialPoint, velocityPoint);
//            } else {
//                value = calculateDistributionFunction(spatialPoint, velocityPoint);
//            }
//            curr->setValue(spatialPoint, velocityPoint, value);
            }
        }
    };

    virtual double calculateFlowFactor(const SpatialPoint& p) = 0;
    virtual double calculateDistributionFunction(const SpatialPoint& p, const Velocity& v) = 0;
};


#endif //CALC_EVOLUTION_H
