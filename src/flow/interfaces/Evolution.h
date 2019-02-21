//
// Created by geoolekom on 20.02.19.
//

#ifndef CALC_EVOLUTION_H
#define CALC_EVOLUTION_H

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
    typedef typename StateType::spatialPointType spatialPointType;
    typedef typename StateType::velocityType velocityType;

    Evolution (StateType** initialState) : state(initialState) {
        long int size = (*initialState)->getSize();
        double* data = new double[size]();
        prev = new StateType(data, size);
        curr = *state;
    }

    ~Evolution() {
        delete[] prev->getData();
        delete prev;
    }

    void evolve(int numSteps) {
        StateType *temp;
        for (int i = 0; i < numSteps; i ++) {
            std::cout << "Шаг " << i << "\n";
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

        for (spatialPointType& point : *prev) {

            flowFactor = calculateFlowFactor(point);
//        mirrorNormal = geometry->getMirrorNormal(spatialPoint);
//        borderReached = geometry->isBorderReached(spatialPoint);

            for (velocityType& velocity : point) {
                std::cout << velocity.toString() << "\n";
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

    virtual double calculateFlowFactor(const spatialPointType& p) = 0;
    virtual double calculateDistributionFunction(const spatialPointType& p, const velocityType& v) = 0;
};


#endif //CALC_EVOLUTION_H
