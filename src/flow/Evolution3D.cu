//
// Created by geoolekom on 27.04.19.
//

#include <math.h>
#include "Evolution3D.h"
#include "utils/cuda.h"

Evolution3D::Evolution3D(double tStep, State3D *state, Grid3D *grid, IndexTankWithScreen2D *geometry, DoduladCI* ci) :
    tStep(tStep), grid(grid), geometry(geometry), state(state), ci(ci) {

    cudaMallocManaged((void**) &curr, sizeof(State3D), cudaMemAttachGlobal);
    cudaMemcpy(curr, state, sizeof(State3D), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &currData, sizeof(double) * curr->getSize(), cudaMemAttachGlobal);
    curr->cudaSetData(currData);

    cudaMallocManaged((void**) &prev, sizeof(State3D), cudaMemAttachGlobal);
    cudaMemcpy(prev, state, sizeof(State3D), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &prevData, sizeof(double) * prev->getSize(), cudaMemAttachGlobal);
    cudaMemcpy(prevData, state->getData(), sizeof(double) * prev->getSize(), cudaMemcpyHostToDevice);
    prev->cudaSetData(prevData);
};

Evolution3D::~Evolution3D() {
    cudaFree(currData);
    cudaFree(prevData);
    cudaFree(curr);
    cudaFree(prev);
};

State3D* Evolution3D::getState() const {
    return this->curr;
};

void Evolution3D::swap() {
    std::swap(curr, prev);
}

void Evolution3D::exportToHost() {
    cudaMemcpy(state->getData(), curr->getData(), sizeof(double) * curr->getSize(), cudaMemcpyDeviceToHost);
}

__device__ double Evolution3D::limiter(double theta) {
    return fmax(0.0, fmin(2 * theta, fmin((1.0 + theta) / 2, 2.0)));
};

__device__ double Evolution3D::limitValue(double gamma, const intVector &direction, const intVector &xIndex,
                               const intVector &vIndex) {
    double value, thetaNom, result;
    double thetaDenom = prev->getValue(xIndex, vIndex) - prev->getValue(xIndex - direction, vIndex);
    if (gamma > 0) {
        thetaNom = prev->getValue(xIndex - direction, vIndex) - prev->getValue(xIndex - 2 * direction, vIndex);
        value = prev->getValue(xIndex - direction, vIndex);
        result = value + (1 - gamma) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
    } else {
        thetaNom = prev->getValue(xIndex + direction, vIndex) - prev->getValue(xIndex, vIndex);
        value = prev->getValue(xIndex, vIndex);
        result = value - (1 + gamma) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
    }
    return fmax(result, 0.);
};

__device__ double Evolution3D::schemeChange(const intVector &xIndex, int vxIndex, int vyIndex, int vzIndex) {
    intVector vIndex = {vxIndex, vyIndex, vzIndex};
    double gammaX = grid->getVx(vxIndex) * tStep / grid->xStep;
    double gammaY = grid->getVy(vyIndex) * tStep / grid->yStep;

    return prev->getValue(xIndex, vIndex)
           - gammaX * (limitValue(gammaX, {1, 0, 0}, xIndex + intVector({1, 0, 0}), vIndex) - limitValue(gammaX, {1, 0, 0}, xIndex, vIndex))
           - gammaY * (limitValue(gammaY, {0, 1, 0}, xIndex + intVector({0, 1, 0}), vIndex) - limitValue(gammaY, {0, 1, 0}, xIndex, vIndex));
};

__device__ void Evolution3D::makeStep(int step, int txIndex, int tyIndex, int txStep, int tyStep) {
    double h, value;
    intVector xIndex, vIndex;
    doubleVector v;

    for (int xzIndex = 0; xzIndex < prev->zIndexMax; xzIndex += 1) {
        for (int xyIndex = tyIndex; xyIndex < prev->yIndexMax; xyIndex += tyStep) {
            for (int xxIndex = txIndex; xxIndex < prev->xIndexMax; xxIndex += txStep) {
                xIndex = {xxIndex, xyIndex, xzIndex};

                h = 0;
                if (geometry->isDiffuseReflection(xIndex, {1, 0, 0})) {
                    h = calculateDiffusionFactor(xIndex, {1, 0, 0});
                } else if (geometry->isDiffuseReflection(xIndex, {-1, 0, 0})) {
                    h = calculateDiffusionFactor(xIndex, {-1, 0, 0});
                } else if (geometry->isDiffuseReflection(xIndex, {0, -1, 0})) {
                    h = calculateDiffusionFactor(xIndex, {0, -1, 0});
                } else if (geometry->isDiffuseReflection(xIndex, {0, 1, 0})) {
                    h = calculateDiffusionFactor(xIndex, {0, 1, 0});
                }
//                printf("PREV: %d %d %d, %f\n", xxIndex, xyIndex, xzIndex, prev->getVelocitySlice(xIndex)[0]);

                for (int vzIndex = prev->vzIndexMin; vzIndex < prev->vzIndexMax; vzIndex ++) {
                    for (int vyIndex = prev->vyIndexMin; vyIndex < prev->vyIndexMax; vyIndex++) {
                        for (int vxIndex = prev->vxIndexMin; vxIndex < prev->vxIndexMax; vxIndex++) {
                            vIndex = {vxIndex, vyIndex, vzIndex};
                            v = grid->getV(vIndex);

                            if (grid->inBounds(v)) {
                                if (geometry->isDiffuseReflection(xIndex, v)) {
                                    value = h * exp(-(v * v) / 2);
                                } else if (geometry->isMirrorReflection(xIndex, v)) {
                                    doubleVector mirroredV = {grid->getVx(vxIndex), - grid->getVy(vyIndex), grid->getVz(vzIndex)};
                                    auto mirroredIndex = grid->getVIndex(mirroredV);
                                    value = prev->getValue(xIndex, mirroredIndex);
                                } else if (geometry->isBorderReached(xIndex, v)) {
                                    value = prev->getValue(xIndex, vIndex);
                                } else {
                                    value = this->schemeChange(xIndex, vxIndex, vyIndex, vzIndex);
                                }
                            } else {
                                value = prev->getValue(xIndex, vIndex);
                            }
                            curr->setValue(xIndex, vIndex, value);
                        }
                    }
                }

//                printf("CURR: %d %d %d, %f\n", xxIndex, xyIndex, xzIndex, curr->getVelocitySlice(xIndex)[0]);
                ci->calculateIntegral(curr->getVelocitySlice(xIndex));
            }
        }
    }
}


__device__ double  Evolution3D::calculateDiffusionFactor(const intVector& xIndex, const intVector& direction) {
    double denom = 0, nom = 0, multiplier;
    doubleVector v;
    intVector vIndex;

    for (int vzIndex = prev->vzIndexMin; vzIndex < prev->vzIndexMax; vzIndex ++) {
        for (int vxIndex = prev->vxIndexMin; vxIndex < prev->vxIndexMax; vxIndex++) {
            for (int vyIndex = prev->vyIndexMin; vyIndex < prev->vyIndexMax; vyIndex++) {
                vIndex = {vxIndex, vyIndex, vzIndex};
                if (grid->inBounds(vIndex)) {
                    v = grid->getV(vIndex);
                    multiplier = v * direction;
                    if (multiplier > 0) {
                        denom += multiplier * exp(-(v * v) / 2);
                    } else if (multiplier < 0) {
                        nom += multiplier *
                               (prev->getValue(xIndex, vIndex) + prev->getValue(xIndex + direction, vIndex)) / 2.0;
                    }
                }
            }
        }
    }
    return denom == 0 ? 0 : - nom / denom;
}
