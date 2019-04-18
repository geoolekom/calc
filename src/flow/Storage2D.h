//
// Created by geoolekom on 23.12.18.
//

#ifndef CALC_STORAGE2D_H
#define CALC_STORAGE2D_H


#include <ostream>
#include <iostream>
#include <cmath>
#include "State2D.h"
#include "Grid2D.h"

class Storage2D {
private:
    State2D* state;
    Grid2D* grid;
public:
    Storage2D() = default;
    Storage2D(State2D* state, Grid2D* grid) : state(state), grid(grid) {};

    void exportDensity(std::ostream* stream) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex ++) {
            for (int yIndex = 0; yIndex < state->yIndexMax; yIndex ++) {
                double value = this->getDensity(xIndex, yIndex);
                (*stream)
                        << grid->getX(xIndex) << "\t"
                        << grid->getY(yIndex) << "\t"
                        << this->getDensity(xIndex, yIndex) << "\n";
            }
            (*stream) << std::endl;
        }
    }

    void exportTemperature(std::ostream* stream) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex ++) {
            for (int yIndex = 0; yIndex < state->yIndexMax; yIndex ++) {
                (*stream)
                        << grid->getX(xIndex) << "\t"
                        << grid->getY(yIndex) << "\t"
                        << this->getTemperature({xIndex, yIndex}) << "\n";
            }
            (*stream) << std::endl;
        }
    }

    void exportTemperatureTensor(std::ostream* stream, const intVector& tensorIndex) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex ++) {
            for (int yIndex = 0; yIndex < state->yIndexMax; yIndex ++) {
                (*stream)
                        << grid->getX(xIndex) << "\t"
                        << grid->getY(yIndex) << "\t"
                        << this->getTemperatureTensor({xIndex, yIndex}, tensorIndex) << "\n";
            }
            (*stream) << std::endl;
        }
    }


    void exportFunction(std::ostream* stream, int xIndex, int yIndex) {
        doubleVector v;
        double value = 0;
        for (int vyIndex = state->vyIndexMin; vyIndex < state->vyIndexMax; vyIndex ++) {
            for (int vxIndex = state->vxIndexMin; vxIndex < state->vxIndexMax; vxIndex ++) {
                v = grid->getV({vxIndex, vyIndex});
                if (grid->inBounds(v)) {
                    value = state->getValue(xIndex, yIndex, vxIndex, vyIndex);
                } else {
                    value = 0;
                }
                (*stream) << v[0] << "\t" << v[1] << "\t" << value << "\n";
            }
            (*stream) << std::endl;
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

    void exportVelocity(std::ostream* stream) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex ++) {
            for (int yIndex = 0; yIndex < state->yIndexMax; yIndex ++) {
                auto value = this->getVelocity({xIndex, yIndex});
                (*stream)
                        << grid->getX(xIndex) << "\t"
                        << grid->getY(yIndex) << "\t"
                        << value[0] << "\t" << value[1] << "\n";
            }
            (*stream) << std::endl;
        }
    }

    void exportRadius(std::ostream* stream) {
        const double k = 0.5;

        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex ++) {
            const double axisValue = this->getDensity(xIndex, 0);
            bool radiusFound = false;
            for (int yIndex = 1; yIndex < state->yIndexMax; yIndex ++) {
                auto value = this->getDensity(xIndex, yIndex);
                if (value < axisValue * k) {
                    (*stream) << grid->getX(xIndex) << "\t" << grid->getY(yIndex) << "\n";
                    radiusFound = true;
                    break;
                }
            }
            if (!radiusFound) {
                (*stream) << grid->getX(xIndex) << "\t" << 0 << "\n";
            }
        }
    }

    void exportMachNumber(std::ostream* stream, int yIndex) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex ++) {
            auto axisV = this->getVelocity({xIndex, yIndex});
            auto axisT = this->getTemperature({xIndex, yIndex});
            (*stream) << grid->getX(xIndex) << "\t" << axisV[0] / std::sqrt(5 * axisT / 3) << "\n";
        }
    }


    double getDensity(int xIndex, int yIndex) {
        double value = 0, vx, vy;
        doubleVector v;
        for (const auto& vIndex : state->getVelocityIterable()) {
            v = grid->getV(vIndex);
            if (grid->inBounds(v)) {
                value += state->getValue({xIndex, yIndex}, vIndex);
            }
        }
        return value;
    }

    double getTemperature(const intVector& xIndex) {
        double nom = 0, denom = 0;
        doubleVector v;
        const auto vAvg = this->getVelocity(xIndex);

        for (const auto& vIndex : state->getVelocityIterable()) {
            v = grid->getV(vIndex);
            if (grid->inBounds(v)) {
                denom += state->getValue(xIndex, vIndex);
                nom += (v - vAvg) * (v - vAvg) * state->getValue(xIndex, vIndex);
            }
        }

        return (nom / denom + 1) / 3;
    }

    double getTemperatureTensor(const intVector& xIndex, const intVector& tensorIndex) {
        double nom = 0, denom = 0;
        doubleVector v;
        const auto vAvg = this->getVelocity(xIndex);

        for (const auto& vIndex : state->getVelocityIterable()) {
            v = grid->getV(vIndex);
            if (grid->inBounds(v)) {
                denom += state->getValue(xIndex, vIndex);
                nom += (v - vAvg)[tensorIndex[0]] * (v - vAvg)[tensorIndex[1]] * state->getValue(xIndex, vIndex);
            }
        }

        return nom / denom;
    }

    double getEnergy(const intVector& xIndex) {
        doubleVector v;
        double nom = 0, denom = 0;
        for (const auto& vIndex : state->getVelocityIterable()) {
            v = grid->getV(vIndex);
            if (grid->inBounds(v)) {
                denom += state->getValue(xIndex, vIndex);
                nom += (v * v) * state->getValue(xIndex, vIndex);
            }
        }
        return nom / denom;
    }

    doubleVector getVelocity(const intVector& xIndex) {
        doubleVector v, vAvg = {0, 0};
        double nom = 0, denom = 0, vx, vy, vxAvg = 0, vyAvg = 0;
        for (const auto& vIndex : state->getVelocityIterable()) {
            v = grid->getV(vIndex);
            if (grid->inBounds(v)) {
                denom += state->getValue(xIndex, vIndex);
                vAvg = vAvg + state->getValue(xIndex, vIndex) * v;
            }
        }
        return vAvg / denom;
    }
};


#endif //CALC_STORAGE2D_H
