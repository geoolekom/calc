//
// Created by geoolekom on 27.04.19.
//

#include <math.h>
#include "Evolution3D.h"
#include "utils/cuda.h"

Evolution3D::Evolution3D(double tStep, State3D *state, Grid3D *grid, RoundHoleTank *geometry, DoduladCI *ci)
    : tStep(tStep), grid(grid), geometry(geometry), state(state), ci(ci) {

    cudaMallocManaged((void**) &curr, sizeof(State3D), cudaMemAttachGlobal);
    cudaMemcpy(curr, state, sizeof(State3D), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &currData, sizeof(floatType) * curr->getSize(), cudaMemAttachGlobal);
    curr->cudaSetData(currData);

    cudaMallocManaged((void**) &prev, sizeof(State3D), cudaMemAttachGlobal);
    cudaMemcpy(prev, state, sizeof(State3D), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &prevData, sizeof(floatType) * prev->getSize(), cudaMemAttachGlobal);
    cudaMemcpy(prevData, state->getData(), sizeof(floatType) * prev->getSize(), cudaMemcpyHostToDevice);
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
    cudaMemcpy(state->getData(), curr->getData(), sizeof(floatType) * curr->getSize(), cudaMemcpyDeviceToHost);
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
    return result;
};

__device__ double Evolution3D::schemeChange(const intVector &xIndex, int vxIndex, int vyIndex, int vzIndex) {
    intVector vIndex = {vxIndex, vyIndex, vzIndex};
    double gammaX = grid->getVx(vxIndex) * tStep / grid->xStep;
    double gammaY = grid->getVy(vyIndex) * tStep / grid->yStep;

    return prev->getValue(xIndex, vIndex)
           - gammaX * (limitValue(gammaX, {1, 0, 0}, xIndex + intVector({1, 0, 0}), vIndex) - limitValue(gammaX, {1, 0, 0}, xIndex, vIndex))
           - gammaY * (limitValue(gammaY, {0, 1, 0}, xIndex + intVector({0, 1, 0}), vIndex) - limitValue(gammaY, {0, 1, 0}, xIndex, vIndex));
};

__device__ void Evolution3D::makeStep(int step, int txIndex, int tyIndex, int tzIndex, int txStep, int tyStep, int tzStep) {
    double h, value, gamma, directionStep;
    bool isMirrored, isBorder;
    intVector xIndex, vIndex, direction;
    doubleVector v, doubleDirection;

    if (step % 3 == 0) {
        direction = {1, 0, 0};
        doubleDirection = {1., 0, 0};
        directionStep = grid->xStep;
    } else if (step % 3 == 1) {
        direction = {0, 1, 0};
        doubleDirection = {0, 1., 0};
        directionStep = grid->yStep;
    } else {
        direction = {0, 0, 1};
        doubleDirection = {0, 0, 1.};
        directionStep = grid->zStep;
    }

    for (int xzIndex = tzIndex; xzIndex < prev->zIndexMax; xzIndex += tzStep) {
        for (int xyIndex = tyIndex; xyIndex < prev->yIndexMax; xyIndex += tyStep) {
            for (int xxIndex = txIndex; xxIndex < prev->xIndexMax; xxIndex += txStep) {
                xIndex = {xxIndex, xyIndex, xzIndex};

                for (int vzIndex = prev->vzIndexMin; vzIndex < prev->vzIndexMax; vzIndex ++) {
                    for (int vyIndex = prev->vyIndexMin; vyIndex < prev->vyIndexMax; vyIndex++) {
                        for (int vxIndex = prev->vxIndexMin; vxIndex < prev->vxIndexMax; vxIndex++) {
                            vIndex = {vxIndex, vyIndex, vzIndex};
                            v = grid->getV(vIndex);

                            if (grid->inBounds(v) && geometry->isFreeFlow(xIndex, v)) {
                                gamma = (v * direction) * tStep / directionStep;
                                value = prev->getValue(xIndex, vIndex) -
                                        gamma * (limitValue(gamma, direction, xIndex + direction, vIndex) -
                                                 limitValue(gamma, direction, xIndex, vIndex));
                                if (value < 0) {
                                    printf("Ошибка: (%d, %d, %d), (%d, %d, %d)\t%.20f\n",
                                           xxIndex, xyIndex, xzIndex, vxIndex, vyIndex, vzIndex, value);
                                    value = prev->getValue(xIndex, vIndex);
                                }
                                curr->setValue(xIndex, vIndex, value);
                            }
                        }
                    }
                }

                if (geometry->isDiffuseReflection(xIndex, doubleDirection)) {
                    h = this->calculateDiffusionFactor(xIndex, direction);
                } else if (geometry->isDiffuseReflection(xIndex, - 1 * doubleDirection)) {
                    h = this->calculateDiffusionFactor(xIndex, - 1 * direction);
                } else {
                    h = 0;
                }

                isMirrored = geometry->isMirrorReflection(xIndex, doubleDirection) || geometry->isMirrorReflection(xIndex, - 1 * doubleDirection);
                isBorder = geometry->isBorderReached(xIndex, doubleDirection) || geometry->isBorderReached(xIndex, - 1 * doubleDirection);

                for (int vzIndex = prev->vzIndexMin; vzIndex < prev->vzIndexMax; vzIndex ++) {
                    for (int vyIndex = prev->vyIndexMin; vyIndex < prev->vyIndexMax; vyIndex++) {
                        for (int vxIndex = prev->vxIndexMin; vxIndex < prev->vxIndexMax; vxIndex++) {
                            vIndex = {vxIndex, vyIndex, vzIndex};
                            v = grid->getV(vIndex);

                            if (grid->inBounds(v)) {
                                if (geometry->isDiffuseReflection(xIndex, v) && h != 0) {
                                    value = h * exp(-(v * v) / 2);
                                } else if (geometry->isMirrorReflection(xIndex, v) && isMirrored) {
                                    doubleVector mirroredV = v - 2 * (doubleDirection * v) * doubleDirection;
                                    auto mirroredIndex = grid->getVIndex(mirroredV);
                                    value = curr->getValue(xIndex, mirroredIndex);
                                } else if (geometry->isBorderReached(xIndex, v) && isBorder) {
                                    value = prev->getValue(xIndex, vIndex);
                                } else if (geometry->isFreeFlow(xIndex, v)) {
                                    value = curr->getValue(xIndex, vIndex);
                                } else {
                                    value = prev->getValue(xIndex, vIndex);
                                }
                            } else {
                                value = prev->getValue(xIndex, vIndex);
                            }
                            curr->setValue(xIndex, vIndex, value);
                        }
                    }
                }
                if (step % 3 == 2) { // Применяем интеграл только после прохода по всем направлениям
                    ci->calculateIntegral(curr->getVelocitySlice(xIndex));
                }
            }
        }
    }
}


__device__ double  Evolution3D::calculateDiffusionFactor(const intVector& xIndex, const intVector& direction) {
    double denom = 0, nom = 0, multiplier;
    doubleVector v;
    intVector vIndex;

    for (int vzIndex = curr->vzIndexMin; vzIndex < curr->vzIndexMax; vzIndex ++) {
        for (int vxIndex = curr->vxIndexMin; vxIndex < curr->vxIndexMax; vxIndex++) {
            for (int vyIndex = curr->vyIndexMin; vyIndex < curr->vyIndexMax; vyIndex++) {
                vIndex = {vxIndex, vyIndex, vzIndex};
                if (grid->inBounds(vIndex)) {
                    v = grid->getV(vIndex);
                    multiplier = v * direction;
                    if (multiplier > 0) {
                        denom += multiplier * exp(-(v * v) / 2);
                    } else if (multiplier < 0) {
                        nom += multiplier *
                               (curr->getValue(xIndex, vIndex) + curr->getValue(xIndex + direction, vIndex)) / 2.0;
                    }
                }
            }
        }
    }
    return denom == 0 ? 0 : - nom / denom;
}
