//
// Created by geoolekom on 21.12.18.
//

#ifndef CALC_EVOLUTION2D_H
#define CALC_EVOLUTION2D_H


#include <stdio.h>
#include <math.h>
#include <host_defines.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "Grid2D.cu"
#include "IndexTankWithScreen2D.cu"
#include "IndexTankFull2D.h"
#include "State2D.cu"
#include "interfaces/CollisionIntegral.h"


class Evolution2D {
public:
    IndexTankWithScreen2D* geometry;
    Grid2D* grid;

    State2D* returningState;
    State2D* curr;
    State2D* prev;

    // Data on device
    double* currData;
    double* prevData;

    double tStep;
    CollisionIntegral<State2D>* ci;

public:
    Evolution2D(double tStep, State2D* state, Grid2D* grid, IndexTankWithScreen2D* geometry, CollisionIntegral<State2D>* ci) :
            returningState(state), grid(grid), geometry(geometry), tStep(tStep), ci(ci) {
        cudaMallocManaged((void**) &curr, sizeof(State2D), cudaMemAttachGlobal);
        cudaMemcpy(curr, returningState, sizeof(State2D), cudaMemcpyHostToDevice);
        cudaMalloc((void**) &currData, sizeof(double) * curr->getSize());
        curr->setData(currData);

        cudaMallocManaged((void**) &prev, sizeof(State2D), cudaMemAttachGlobal);
        cudaMemcpy(prev, returningState, sizeof(State2D), cudaMemcpyHostToDevice);
        cudaMalloc((void**) &prevData, sizeof(double) * curr->getSize());
        cudaMemcpy(prevData, returningState->getData(), sizeof(double) * prev->getSize(), cudaMemcpyHostToDevice);
        prev->setData(prevData);
    }

    ~Evolution2D() {
        cudaFree(curr);
        cudaFree(prev);
        cudaFree(currData);
        cudaFree(prevData);
    }

    void swap() {
        std::swap(curr, prev);
    }

    void exportToHost() {
        cudaMemcpy(returningState->getData(), curr->getData(), sizeof(double) * curr->getSize(), cudaMemcpyDeviceToHost);
    }

    __device__ double calculateDiffusionFactor(const intVector& xIndex, const intVector& direction) {
        double denom = 0, nom = 0, multiplier;
        doubleVector v;
        intVector vIndex;

        for (int vxIndex = prev->vxIndexMin; vxIndex < prev->vxIndexMax; vxIndex ++) {
            for (int vyIndex = prev->vyIndexMin; vyIndex < prev->vyIndexMax; vyIndex ++) {
                vIndex = {vxIndex, vyIndex};
                v = grid->getV(vIndex);
                if (grid->inBounds(v)) {
                    multiplier = v * direction;
                    if (multiplier > 0) {
                        denom += multiplier * exp(- (v * v) / 2);
                    } else if (multiplier < 0) {
                        nom += multiplier * (prev->getValue(xIndex, vIndex) + prev->getValue(xIndex + direction, vIndex)) / 2.0;
                    }
                }
            }
        }
        return denom == 0 ? 0 : - nom / denom;
    }

    __device__ void makeStep(int step, int txIndex, int txStep, int tyIndex, int tyStep) {
        double h, value;
        intVector xIndex, vIndex;
        doubleVector v;

//        ci->stepForward();

        for (int xyIndex = tyIndex; xyIndex < prev->yIndexMax; xyIndex += tyStep) {
            for (int xxIndex = txIndex; xxIndex < prev->xIndexMax; xxIndex += txStep) {
                xIndex = {xxIndex, xyIndex};
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

                for (int vxIndex = prev->vxIndexMin; vxIndex < prev->vxIndexMax; vxIndex ++) {
                    for (int vyIndex = prev->vyIndexMin; vyIndex < prev->vyIndexMax; vyIndex ++) {
                        vIndex = {vxIndex, vyIndex};
                        v = grid->getV(vIndex);

                        if (grid->inBounds(v)) {
                            if (geometry->isDiffuseReflection(xIndex, v)) {
                                value = h * exp(-(v * v) / 2);
                            } else if (geometry->isMirrorReflection(xIndex, v)) {
                                int mirroredYIndex = grid->getVyIndex(- grid->getVy(vyIndex));
                                value = prev->getValue(xxIndex, xyIndex, vxIndex, mirroredYIndex);
                            } else if (geometry->isBorderReached(xIndex, v)) {
                                value = prev->getValue(xxIndex, xyIndex, vxIndex, vyIndex);
                            } else {
                                value = this->schemeChange(xxIndex, xyIndex, vxIndex, vyIndex);
                            }
                            curr->setValue(xxIndex, xyIndex, vxIndex, vyIndex, value);
                        }
                    }
                }

//                ci->calculateIntegral(curr, xxIndex, xyIndex);
            }
        }
    }

    __device__ inline double limiter(double theta) {
        return fmax(0.0, fmin(2 * theta, fmin((1.0 + theta) / 2, 2.0)));
        // return std::max(0.0, std::min(1.0, theta));
    }

    __device__ inline double limitValueX(double gammaX, int xIndex, int yIndex, int vxIndex, int vyIndex) {
        double value, thetaNom, result;
        double thetaDenom = prev->getValue(xIndex, yIndex, vxIndex, vyIndex) - prev->getValue(xIndex - 1, yIndex, vxIndex, vyIndex);
        if (gammaX > 0) {
            thetaNom = prev->getValue(xIndex - 1, yIndex, vxIndex, vyIndex) - prev->getValue(xIndex - 2, yIndex, vxIndex, vyIndex);
            value = prev->getValue(xIndex - 1, yIndex, vxIndex, vyIndex);
            result = value + (1 - gammaX) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
        } else {
            thetaNom = prev->getValue(xIndex + 1, yIndex, vxIndex, vyIndex) - prev->getValue(xIndex, yIndex, vxIndex, vyIndex);
            value = prev->getValue(xIndex, yIndex, vxIndex, vyIndex);
            result = value - (1 + gammaX) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
        }
        return fmax(result, 0.);
    }

    __device__ inline double limitValueY(double gammaY, int xIndex, int yIndex, int vxIndex, int vyIndex) {
        double value, thetaNom, result;
        double thetaDenom = prev->getValue(xIndex, yIndex, vxIndex, vyIndex) - prev->getValue(xIndex, yIndex - 1, vxIndex, vyIndex);
        if (gammaY > 0) {
            thetaNom = prev->getValue(xIndex, yIndex - 1, vxIndex, vyIndex) - prev->getValue(xIndex, yIndex - 2, vxIndex, vyIndex);
            value = prev->getValue(xIndex, yIndex - 1, vxIndex, vyIndex);
            result = value + (1 - gammaY) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
        } else {
            thetaNom = prev->getValue(xIndex, yIndex + 1, vxIndex, vyIndex) - prev->getValue(xIndex, yIndex, vxIndex, vyIndex);
            value = prev->getValue(xIndex, yIndex, vxIndex, vyIndex);
            result = value - (1 + gammaY) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
        }
        return fmax(result, 0.);
    }

    __device__ double schemeChange(int xIndex, int yIndex, int vxIndex, int vyIndex) {
        double gammaX = grid->getVx(vxIndex) * tStep / grid->xStep;
        double gammaY = grid->getVy(vyIndex) * tStep / grid->yStep;

        return prev->getValue(xIndex, yIndex, vxIndex, vyIndex)
               - gammaX * (limitValueX(gammaX, xIndex + 1, yIndex, vxIndex, vyIndex) - limitValueX(gammaX, xIndex, yIndex, vxIndex, vyIndex))
               - gammaY * (limitValueY(gammaY, xIndex, yIndex + 1, vxIndex, vyIndex) - limitValueY(gammaY, xIndex, yIndex, vxIndex, vyIndex));
    }
};


#endif //CALC_EVOLUTION2D_H
