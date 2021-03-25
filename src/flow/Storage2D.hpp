//
// Created by geoolekom on 23.12.18.
//

#ifndef CALC_STORAGE2D_HPP
#define CALC_STORAGE2D_HPP

#include "Grid3D.h"
#include "State3D.h"
#include <cmath>
#include <iostream>
#include <ostream>

class Storage2D {
  private:
    State3D *state;
    Grid3D *grid;

  public:
    Storage2D() = default;
    Storage2D(State3D *state, Grid3D *grid) : state(state), grid(grid){};

    void importState(std::ifstream *stream) {
        intVector fileXIndex, fileVIndex;
        double value;
        for (const auto &xIndex : state->getSpaceIterable()) {
            for (const auto &vIndex : state->getVelocityIterable()) {
                (*stream) >> fileXIndex[0] >> fileXIndex[1] >> fileXIndex[2] >> fileVIndex[0] >> fileVIndex[1] >>
                    fileVIndex[2] >> value;
                if (fileXIndex != xIndex || fileVIndex != vIndex) {
                    throw std::logic_error("Ошибка формата ввода");
                }
                state->setValue(xIndex, vIndex, value);
            }
        }
    }

    void exportState(std::ofstream *stream) {
        doubleVector v;
        double value;
        for (const auto &xIndex : state->getSpaceIterable()) {
            for (const auto &vIndex : state->getVelocityIterable()) {
                v = grid->getV(vIndex);
                if (grid->inBounds(v)) {
                    value = state->getValue(xIndex, vIndex);
                } else {
                    value = 0;
                }
                (*stream) << xIndex[0] << "\t" << xIndex[1] << "\t" << xIndex[2] << "\t" << vIndex[0] << "\t"
                          << vIndex[1] << "\t" << vIndex[2] << "\t" << value << "\n";
                ;
            }
        }
    }

    void exportAll(std::ostream *stream) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex++) {
            for (int yIndex = 0; yIndex < state->yIndexMax; yIndex++) {
                for (int zIndex = 0; zIndex < state->zIndexMax; zIndex++) {
                    auto density = this->getDensity(xIndex, yIndex, zIndex);
                    auto temperature = this->getTemperature({xIndex, yIndex, zIndex});
                    auto velocity = this->getVelocity({xIndex, yIndex, zIndex});
                    (*stream) << grid->getXx(xIndex) << "\t" << grid->getXy(yIndex) << "\t" << grid->getXz(zIndex)
                              << "\t" << density << "\t" << temperature << "\t" << velocity[0] << "\t" << velocity[1]
                              << "\t" << velocity[2] << "\t"
                              << "\t" << std::endl;
                }
                (*stream) << std::endl;
            }
            (*stream) << std::endl;
        }
    }

    void exportDensity(std::ostream *stream) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex++) {
            for (int yIndex = 0; yIndex < state->yIndexMax; yIndex++) {
                (*stream) << grid->getXx(xIndex) << "\t" << grid->getXy(yIndex) << "\t"
                          << this->getDensity(xIndex, yIndex, 0) << "\n";
            }
            (*stream) << std::endl;
        }
    }

    void exportTemperature(std::ostream *stream) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex++) {
            for (int yIndex = 0; yIndex < state->yIndexMax; yIndex++) {
                (*stream) << grid->getXx(xIndex) << "\t" << grid->getXy(yIndex) << "\t"
                          << this->getTemperature({xIndex, yIndex, 0}) << "\n";
            }
            (*stream) << std::endl;
        }
    }

    void exportTemperatureTensor(std::ostream *stream, const intVector &tensorIndex) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex++) {
            for (int yIndex = 0; yIndex < state->yIndexMax; yIndex++) {
                (*stream) << grid->getXx(xIndex) << "\t" << grid->getXy(yIndex) << "\t"
                          << this->getTemperatureTensor({xIndex, yIndex, 0}, tensorIndex) << "\n";
            }
            (*stream) << std::endl;
        }
    }

    void exportFunction(std::ostream *stream, int xIndex, int yIndex, int zIndex) {
        doubleVector v;
        double value;
        for (int vzIndex = state->vzIndexMin; vzIndex < state->vzIndexMax; vzIndex++) {
            for (int vyIndex = state->vyIndexMin; vyIndex < state->vyIndexMax; vyIndex++) {
                for (int vxIndex = state->vxIndexMin; vxIndex < state->vxIndexMax; vxIndex++) {
                    v = grid->getV({vxIndex, vyIndex, vzIndex});
                    if (grid->inBounds(v)) {
                        value = state->getValue({xIndex, yIndex, zIndex}, {vxIndex, vyIndex, vzIndex});
                    } else {
                        value = 0;
                    }
                    (*stream) << v[0] << "\t" << v[1] << "\t" << v[2] << "\t" << value << "\n";
                }
                (*stream) << std::endl;
            }
        }
    }

    void exportFlowX(std::ostream *stream, int step, int xIndex, int yIndexMax) {
        doubleVector v;
        double value = 0;
        for (int yIndex = 0; yIndex < yIndexMax; yIndex++) {
            for (const auto &vIndex : state->getVelocityIterable()) {
                v = grid->getV(vIndex);
                if (grid->inBounds(v)) {
                    value += v[0] * state->getValue({xIndex, yIndex}, vIndex);
                }
            }
        }
        (*stream) << step << "\t" << value << "\n";
    }

    void exportVelocity(std::ostream *stream) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex++) {
            for (int yIndex = 0; yIndex < state->yIndexMax; yIndex++) {
                auto value = this->getVelocity({xIndex, yIndex});
                (*stream) << grid->getXx(xIndex) << "\t" << grid->getXy(yIndex) << "\t" << value[0] << "\t" << value[1]
                          << "\n";
            }
            (*stream) << std::endl;
        }
    }

    void exportRadius(std::ostream *stream, int holeCenterY, int holeCenterZ) {
        const double k = 0.5;

        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex++) {
            const double axisValue = this->getDensity(xIndex, holeCenterY, holeCenterZ);
            bool radiusFound = false;
            for (int yIndex = holeCenterY + 1; yIndex < state->yIndexMax; yIndex++) {
                auto value = this->getDensity(xIndex, yIndex, holeCenterZ);
                if (value < axisValue * k) {
                    auto prevValue = this->getDensity(xIndex, yIndex - 1, holeCenterZ);
                    (*stream) << grid->getXx(xIndex) << "\t"
                              << grid->getXy(yIndex) - (axisValue * k - value) * grid->yStep / (prevValue - value) -
                                     holeCenterY
                              << "\n";
                    radiusFound = true;
                    break;
                }
            }
            if (!radiusFound) {
                (*stream) << grid->getXx(xIndex) << "\t" << 0 << "\n";
            }
        }
    }

    void exportMachNumber(std::ostream *stream, int yIndex) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex++) {
            auto axisV = this->getVelocity({xIndex, yIndex});
            auto axisT = this->getTemperature({xIndex, yIndex});
            (*stream) << grid->getXx(xIndex) << "\t" << axisV[0] / std::sqrt(5 * axisT / 3) << "\n";
        }
    }

    double getDensity(int xIndex, int yIndex, int zIndex) {
        double value = 0;
        doubleVector v;
        for (const auto &vIndex : state->getVelocityIterable()) {
            v = grid->getV(vIndex);
            if (grid->inBounds(v)) {
                value += state->getValue({xIndex, yIndex, zIndex}, vIndex);
            }
        }
        return value;
    }

    double getTemperature(const intVector &xIndex) {
        double nom = 0;
        double denom = 0;
        doubleVector v;
        const auto vAvg = this->getVelocity(xIndex);

        for (const auto &vIndex : state->getVelocityIterable()) {
            v = grid->getV(vIndex);
            if (grid->inBounds(v)) {
                denom += state->getValue(xIndex, vIndex);
                nom += (v - vAvg) * (v - vAvg) * state->getValue(xIndex, vIndex);
            }
        }

        return (nom / denom) / 3;
    }

    double getTemperatureTensor(const intVector &xIndex, const intVector &tensorIndex) {
        double nom = 0;
        double denom = 0;
        doubleVector v;
        const auto vAvg = this->getVelocity(xIndex);

        for (const auto &vIndex : state->getVelocityIterable()) {
            v = grid->getV(vIndex);
            if (grid->inBounds(v)) {
                denom += state->getValue(xIndex, vIndex);
                nom += (v - vAvg)[tensorIndex[0]] * (v - vAvg)[tensorIndex[1]] * state->getValue(xIndex, vIndex);
            }
        }

        return nom / denom;
    }

    double getEnergy(const intVector &xIndex) {
        doubleVector v;
        double nom = 0;
        double denom = 0;
        for (const auto &vIndex : state->getVelocityIterable()) {
            v = grid->getV(vIndex);
            if (grid->inBounds(v)) {
                denom += state->getValue(xIndex, vIndex);
                nom += (v * v) * state->getValue(xIndex, vIndex);
            }
        }
        return nom / denom;
    }

    doubleVector getVelocity(const intVector &xIndex) {
        doubleVector v, vAvg = {0, 0, 0};
        double denom = 0;
        for (const auto &vIndex : state->getVelocityIterable()) {
            v = grid->getV(vIndex);
            if (grid->inBounds(v)) {
                denom += state->getValue(xIndex, vIndex);
                vAvg = vAvg + state->getValue(xIndex, vIndex) * v;
            }
        }
        return vAvg / denom;
    }
};

#endif // CALC_STORAGE2D_HPP
