//
// Created by geoolekom on 20.02.19.
//

#ifndef CALC_EVOLUTION_H
#define CALC_EVOLUTION_H


#include <iostream>

template <typename State, typename Geometry, typename Grid, typename Point>
class Evolution {
private:
    State *prev, *curr;
    Geometry *geometry;
    Grid *grid;
public:
    void evolve(State** state, int numSteps) {
        State* temp;
        for (int i = 0; i < numSteps; i ++) {
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
        Point mirrorNormal;

        for (const auto& spatialPoint : prev) {

            flowFactor = calculateFlowFactor(spatialPoint);
            mirrorNormal = geometry->getMirrorNormal(spatialPoint);
            borderReached = geometry->isBorderReached(spatialPoint);

            for (const auto& velocityPoint : spatialPoint) {
                if (geometry->isDiffuseReflection(spatialPoint, velocityPoint)) {
                    value = flowFactor * exp(- velocityPoint * velocityPoint / 2);
                } else if (geometry->isMirrorReflection(spatialPoint, velocityPoint)) {
                    value = prev->getValue(spatialPoint, mirrorNormal * velocityPoint);
                } else if (borderReached) {
                    value = prev->getValue(spatialPoint, velocityPoint);
                } else {
                    value = calculateDistributionFunction(spatialPoint, velocityPoint);
                }
                curr->setValue(spatialPoint, velocityPoint, value);
            }
        }
    };

    virtual double calculateFlowFactor(const Point& p) {};

    virtual double calculateDistributionFunction(const Point& p, const Point& v) {};
};


#endif //CALC_EVOLUTION_H
