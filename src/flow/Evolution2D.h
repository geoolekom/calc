//
// Created by geoolekom on 21.12.18.
//

#ifndef CALC_EVOLUTION2D_H
#define CALC_EVOLUTION2D_H


#include <cmath>
#include <iostream>
#include "Grid2D.h"
#include "Tank2D.h"
#include "TankWithScreen2D.h"
#include "IndexTankWithScreen2D.h"
#include "State2D.h"
#include "interfaces/CollisionIntegral.h"


class Evolution2D {
private:
    IndexTankWithScreen2D* geometry;
    Grid2D* grid;

    State2D** returningState;
    State2D* curr;
    State2D* prev;
    int currentStep = 0;
    double tStep;
    CollisionIntegral<State2D>* ci;

public:
    Evolution2D(double tStep, State2D** state, Grid2D* grid, IndexTankWithScreen2D* geometry, CollisionIntegral<State2D>* ci) :
            returningState(state), grid(grid), geometry(geometry), tStep(tStep), ci(ci) {
        curr = *state;
        prev = new State2D(**state);
    }

    ~Evolution2D() {
        delete prev;
    }

    void evolve(int lastStep) {
        if (lastStep < currentStep) {
            std::cout << "Нельзя произвести эволюцию на шаг, который уже был.\n";
            return;
        }
        State2D* temp;
        for (int i = currentStep; i < lastStep; i ++) {
            std::swap(prev, curr);
            this->makeStep(i);
        }
        currentStep = lastStep;
        *returningState = curr;
    };

    double calculateDiffusionFactor(const intVector& xIndex, const intVector& direction) {
        double denom = 0, nom = 0, vx, vy, multiplier;
        doubleVector v;

        for (const auto& vIndex : prev->getVelocityIterable()) {
            v = grid->getV(vIndex);
            if (grid->inBounds(v)) {
                multiplier = v * direction;
                if (multiplier > 0) {
                    denom += multiplier * exp(- (v * v) / 2);
                } else {
                    nom += multiplier * (prev->getValue(xIndex, vIndex) + prev->getValue(xIndex + direction, vIndex)) / 2.0;
                }
            }
        }
        return denom == 0 ? 0 : - nom / denom;
    }

    void makeStep(int step) {
        double h, vx, vy, value;
        bool borderReached;

        ci->stepForward();

        for (const auto& xIndex : prev->getSpaceIterable()) {
            const auto x = grid->getX(xIndex);
            h = 0;
            if (geometry->isDiffuseReflection(xIndex, {1, 0})) {
                h = calculateDiffusionFactor(xIndex, {1, 0});
            } else if (geometry->isDiffuseReflection(xIndex, {-1, 0})) {
                h = calculateDiffusionFactor(xIndex, {-1, 0});
            } else if (geometry->isDiffuseReflection(xIndex, {0, -1})) {
                h = calculateDiffusionFactor(xIndex, {0, -1});
            } else if (geometry->isDiffuseReflection(xIndex, {0, 1})) {
                h = calculateDiffusionFactor(xIndex, {0, 1});
            }

            for (const auto& vIndex : prev->getVelocityIterable()) {
                const auto v = grid->getV(vIndex);
                int vxIndex = vIndex[0], vyIndex = vIndex[1];

                if (grid->inBounds(v)) {
                    if (geometry->isDiffuseReflection(xIndex, v)) {
                        value = h * exp(-(v * v) / 2);
                    } else if (geometry->isMirrorReflection(xIndex, v)) {
                        value = prev->getValue(xIndex[0], xIndex[1], vxIndex, - vyIndex - 1);
                    } else if (geometry->isBorderReached(xIndex, v)) {
                        value = prev->getValue(xIndex[0], xIndex[1], vxIndex, vyIndex);
                    } else {
                        value = schemeChange(xIndex[0], xIndex[1], vxIndex, vyIndex);
                    }
                    curr->setValue(xIndex[0], xIndex[1], vxIndex, vyIndex, value);
                }
            }

            ci->calculateIntegral(curr, xIndex[0], xIndex[1]);

        }
    }

    inline double limiter(double theta) {
        return std::max(0.0, std::min(1.0, theta));
    }

    inline double limitValueX(double gammaX, int xIndex, int yIndex, int vxIndex, int vyIndex) {
        double value, thetaNom;
        double thetaDenom = prev->getValue(xIndex + 1, yIndex, vxIndex, vyIndex) - prev->getValue(xIndex, yIndex, vxIndex, vyIndex);
        if (gammaX > 0) {
            thetaNom = prev->getValue(xIndex, yIndex, vxIndex, vyIndex) - prev->getValue(xIndex - 1, yIndex, vxIndex, vyIndex);
            value = prev->getValue(xIndex, yIndex, vxIndex, vyIndex);
            return value + (1 - gammaX) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
        } else {
            thetaNom = prev->getValue(xIndex + 2, yIndex, vxIndex, vyIndex) - prev->getValue(xIndex + 1, yIndex, vxIndex, vyIndex);
            value = prev->getValue(xIndex + 1, yIndex, vxIndex, vyIndex);
            return value - (1 + gammaX) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
        }
    }

    inline double limitValueY(double gammaY, int xIndex, int yIndex, int vxIndex, int vyIndex) {
        double value, thetaNom;
        double thetaDenom = prev->getValue(xIndex, yIndex + 1, vxIndex, vyIndex) - prev->getValue(xIndex, yIndex, vxIndex, vyIndex);
        if (gammaY > 0) {
            thetaNom = prev->getValue(xIndex, yIndex, vxIndex, vyIndex) - prev->getValue(xIndex, yIndex - 1, vxIndex, vyIndex);
            value = prev->getValue(xIndex, yIndex, vxIndex, vyIndex);
            return value + (1 - gammaY) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
        } else {
            thetaNom = prev->getValue(xIndex, yIndex + 2, vxIndex, vyIndex) - prev->getValue(xIndex, yIndex + 1, vxIndex, vyIndex);
            value = prev->getValue(xIndex, yIndex + 1, vxIndex, vyIndex);
            return value - (1 + gammaY) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
        }
    }

    double schemeChange(int xIndex, int yIndex, int vxIndex, int vyIndex) {
        double gammaX = grid->getVx(vxIndex) * tStep / grid->xStep;
        double gammaY = grid->getVy(vyIndex) * tStep / grid->yStep;

        return prev->getValue(xIndex, yIndex, vxIndex, vyIndex)
               - gammaX * (limitValueX(gammaX, xIndex, yIndex, vxIndex, vyIndex) - limitValueX(gammaX, xIndex - 1, yIndex, vxIndex, vyIndex))
               - gammaY * (limitValueY(gammaY, xIndex, yIndex, vxIndex, vyIndex) - limitValueY(gammaY, xIndex, yIndex - 1, vxIndex, vyIndex));
    }
};


#endif //CALC_EVOLUTION2D_H
