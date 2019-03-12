//
// Created by geoolekom on 20.02.19.
//

#ifndef CALC_EVOLUTION_H
#define CALC_EVOLUTION_H

#include <cmath>
#include "Space.h"
#include "Geometry.h"
#include "Grid.h"

template <std::size_t dimension>
class Evolution {
public:
    typedef Space<dimension> spaceType;
    typedef typename spaceType::vectorType vectorType;

    typedef Geometry<dimension> geometryType;
    typedef typename geometryType::Matrix matrixType;

    typedef Grid<dimension> gridType;
protected:
    spaceType** space;
    spaceType *prev, *curr;
    geometryType *geometry;
    gridType *grid;
    double tStep;
public:

    Evolution (double tStep, spaceType** initial, geometryType* geometry, gridType* grid) :
        space(initial), geometry(geometry), grid(grid), tStep(tStep) {
        prev = new spaceType(**initial);
        curr = *space;
    }

    ~Evolution() {
        delete prev;
    }

    void evolve(int numSteps) {
        for (int i = 0; i < numSteps; i ++) {
            std::swap(prev, curr);
            this->makeStep(i);
        }
        *space = curr;
    };

    void makeStep(int step) {
        double value, flowFactor;
        bool borderReached;
        matrixType mirrorNormal;
        vectorType diffusionMask;
        vectorType shift = {{5, 5}};

        for (vectorType& xIndex : prev->spaceIterable()) {
            diffusionMask = geometry->getDiffusionMask(xIndex);
            flowFactor = calculateFlowFactor(diffusionMask, xIndex);
            mirrorNormal = geometry->getMirrorNormal(xIndex);
            borderReached = geometry->isBorderReached(xIndex);

            for (vectorType& vIndex : prev->velocityIterable()) {
//                std::cout << "SPACE: " << xIndex[0] << " " << xIndex[1] << "\t";
//                std::cout << "VELOCITY: " << vIndex[0] << " " << vIndex[1] << "\t";
                if (geometry->isDiffuseReflection(xIndex, vIndex - shift)) {
                    auto v = grid->getVelocity(vIndex - shift);
                    value = flowFactor * exp(- (v * v) / 2);
                } else if (geometry->isMirrorReflection(xIndex, vIndex - shift)) {
                    value = prev->getValue(xIndex, mirrorNormal * (vIndex - shift) + shift);
                } else if (borderReached) {
                    value = prev->getValue(xIndex, vIndex);
                } else {
                    value = calculateDistributionFunction(xIndex, vIndex);
                }
                curr->setValue(xIndex, vIndex, value);
//                std::cout << "Value: " << value << "\t" << curr->getValue(xIndex, vIndex) << std::endl;
            }
        }
    };

    double calculateFlowFactor(const vectorType& mask, const vectorType& xIndex) {
        double denom = 0, nom = 0;
        vectorType shift = {{5, 5}};

        for (vectorType& vIndex : prev->velocityIterable()) {
            auto v = grid->getVelocity(vIndex - shift);
            double multiplier = v * mask;
            if (multiplier > 0) {
                denom += multiplier * exp(- (v * v) / 2);
            } else {
                nom += multiplier * (prev->getValue(xIndex, vIndex) + prev->getValue(xIndex + mask, vIndex)) / 2.0;
            }
        }
        return denom == 0 ? 0 : - nom / denom;
    };

    virtual double calculateDistributionFunction(const vectorType& xIndex, const vectorType& vIndex) = 0;
};


#endif //CALC_EVOLUTION_H
