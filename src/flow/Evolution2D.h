//
// Created by geoolekom on 21.12.18.
//

#ifndef CALC_EVOLUTION2D_H
#define CALC_EVOLUTION2D_H


#include <cmath>
#include <iostream>
#include "Grid2D.h"
#include "Tank2D.h"
#include "State2D.h"

class Evolution2D {
private:
    Tank2D* geometry;
    Grid2D* grid;

    State2D** returningState;
    State2D* curr;
    State2D* prev;
    int currentStep = 0;
    double tStep;
public:
    Evolution2D(double tStep, State2D** state, Grid2D* grid, Tank2D* geometry) :
            returningState(state), grid(grid), geometry(geometry), tStep(tStep) {
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
            temp = curr;
            curr = prev;
            prev = temp;
            this->makeStep(i);
        }
        currentStep = lastStep;
        *returningState = curr;
    };

    double calculateDiffusionH(int xIndex, int yIndex, char direction) {
        double denom = 0, nom = 0, vx, vy;
        if (direction == 'L') {
            for (int vyIndex = prev->vyIndexMin; vyIndex < prev->vyIndexMax; vyIndex ++) {
                for (int vxIndex = prev->vxIndexMin; vxIndex < 0; vxIndex++) {
                    vx = grid->getVx(vxIndex);
                    vy = grid->getVy(vyIndex);
                    denom += vx * exp(- (vx * vx + vy * vy) / 2);
                }
            }
            for (int vyIndex = prev->vyIndexMin; vyIndex < prev->vyIndexMax; vyIndex ++) {
                for (int vxIndex = 0; vxIndex < prev->vxIndexMax; vxIndex++) {
                    vx = grid->getVx(vxIndex);
                    nom += vx * (prev->getValue(xIndex, yIndex, vxIndex, vyIndex) + prev->getValue(xIndex + 1, yIndex, vxIndex, vyIndex)) / 2.0;
                }
            }
        } else if (direction == 'R') {
            for (int vyIndex = prev->vyIndexMin; vyIndex < prev->vyIndexMax; vyIndex ++) {
                for (int vxIndex = 0; vxIndex < prev->vxIndexMax; vxIndex++) {
                    vx = grid->getVx(vxIndex);
                    vy = grid->getVy(vyIndex);
                    denom += vx * exp(- (vx * vx + vy * vy) / 2);
                }
            }
            for (int vyIndex = prev->vyIndexMin; vyIndex < prev->vyIndexMax; vyIndex ++) {
                for (int vxIndex = prev->vxIndexMin; vxIndex < 0; vxIndex++) {
                    vx = grid->getVx(vxIndex);
                    nom += vx * (prev->getValue(xIndex - 1, yIndex, vxIndex, vyIndex) + prev->getValue(xIndex, yIndex, vxIndex, vyIndex)) / 2.0;
                }
            }
        } else if (direction == 'U') {
            for (int vyIndex = 0; vyIndex < prev->vyIndexMax; vyIndex ++) {
                for (int vxIndex = prev->vxIndexMin; vxIndex < prev->vxIndexMax; vxIndex++) {
                    vx = grid->getVx(vxIndex);
                    vy = grid->getVy(vyIndex);
                    denom += vy * exp(- (vx * vx + vy * vy) / 2);
                }
            }
            for (int vyIndex = prev->vyIndexMin; vyIndex < 0; vyIndex ++) {
                for (int vxIndex = prev->vxIndexMin; vxIndex < prev->vxIndexMax; vxIndex++) {
                    vy = grid->getVy(vyIndex);
                    nom += vy * (prev->getValue(xIndex, yIndex, vxIndex, vyIndex) + prev->getValue(xIndex, yIndex - 1, vxIndex, vyIndex)) / 2.0;
                }
            }
        } else if (direction == 'D') {
            for (int vyIndex = prev->vyIndexMin; vyIndex < 0; vyIndex ++) {
                for (int vxIndex = prev->vxIndexMin; vxIndex < prev->vxIndexMax; vxIndex++) {
                    vx = grid->getVx(vxIndex);
                    vy = grid->getVy(vyIndex);
                    denom += vy * exp(- (vx * vx + vy * vy) / 2);
                }
            }
            for (int vyIndex = 0; vyIndex < prev->vyIndexMax; vyIndex ++) {
                for (int vxIndex = prev->vxIndexMin; vxIndex < prev->vxIndexMax; vxIndex++) {
                    vy = grid->getVy(vyIndex);
                    nom += vy * (prev->getValue(xIndex, yIndex, vxIndex, vyIndex) + prev->getValue(xIndex, yIndex + 1, vxIndex, vyIndex)) / 2.0;;
                }
            }
        }
        return denom == 0 ? 0 : - nom / denom;
    }

    void makeStep(int step) {
//        int xWallStart = grid->getXIndex(geometry->xWallStart);
//        int xWallEnd = grid->getXIndex(geometry->xWallEnd);
//        int yWallStart = grid->getXIndex(geometry->yWallStart);
//
//        for (int yIndex = 0; yIndex < prev->yIndexMax; yIndex ++) {
//            for (int xIndex = 0; xIndex < prev->xIndexMax; xIndex ++) {
//                if (xIndex == 0) {
//                    // Левая грань
//                    // Диффузия для vx < 0
//                    double diffuseH = calculateDiffusion(xIndex, yIndex);
//                    // Обсчет для vx > 0
//                } else if (xIndex == prev->xIndexMax - 1) {
//                    // Правая грань
//                } else if (yIndex == 0) {
//                    // Нижняя грань
//                    // Зеркальное отражение
//                } else if (yIndex == prev->yIndexMax - 1 && xIndex < xWallStart) {
//                    // Верхняя грань внутри ящика
//                    // Диффузия для vy > 0
//                    // Обсчет для
//                } else if (yIndex == prev->yIndexMax - 1 && xIndex > xWallEnd) {
//                    // Верхняя грань снаружи ящика
//                } else {
//                    return;
//                }
//            }
//        }

        for (int vyIndex = prev->vyIndexMin; vyIndex < prev->vyIndexMax; vyIndex ++) {
            for (int vxIndex = prev->vxIndexMin; vxIndex < prev->vxIndexMax; vxIndex ++) {
                for (int yIndex = 0; yIndex < prev->yIndexMax; yIndex ++) {
                    for (int xIndex = 0; xIndex < prev->xIndexMax; xIndex ++) {
                        double value = vxIndex == 0 && vyIndex == 0 ? 0 : calcValue(xIndex, yIndex, vxIndex, vyIndex);
                        curr->setValue(xIndex, yIndex, vxIndex, vyIndex, value);
                    }
                }
            }
        }
    }

    double calcValue(int xIndex, int yIndex, int vxIndex, int vyIndex) {
        int xWallStart = grid->getXIndex(geometry->xWallStart);
        int xWallEnd = grid->getXIndex(geometry->xWallEnd);
        int yWallStart = grid->getXIndex(geometry->yWallStart);

        bool diffuseReflection =
                (xIndex == 0 && vxIndex < 0) ||
                (yIndex == prev->yIndexMax - 1 && vyIndex > 0 && xIndex <= xWallStart) ||
                (xIndex == xWallStart && vxIndex > 0 && yIndex >= yWallStart) ||
                (xIndex == xWallEnd && vxIndex < 0 && yIndex >= yWallStart);
        if (diffuseReflection) {
            double vx = grid->getVx(vxIndex);
            double vy = grid->getVy(vyIndex);
            double h;
            if (xIndex == 0 || xIndex == xWallEnd) {
                h = calculateDiffusionH(xIndex, yIndex, 'L');
            } else if (xIndex == xWallStart) {
                h = calculateDiffusionH(xIndex, yIndex, 'R');
            } else {
                h = calculateDiffusionH(xIndex, yIndex, 'U');
            }
            return h * exp(- (vx * vx + vy * vy) / 2);
        }
        bool borderReached =
                (xIndex > xWallEnd && yIndex == prev->yIndexMax - 1) ||
                (xIndex == prev->xIndexMax - 1);
        if (borderReached) {
            return 0;
        }
        bool mirrorReflection = (yIndex == 0 && vyIndex < 0);
        if (mirrorReflection) {
            return schemeChange(xIndex, yIndex, vxIndex, vyIndex) - prev->getValue(xIndex, yIndex, vxIndex, vyIndex);
        }
        bool mirrorDestination = (yIndex == 0 && vyIndex > 0);
        if (mirrorDestination) {
            return schemeChange(xIndex, yIndex, vxIndex, vyIndex) + prev->getValue(xIndex, yIndex, vxIndex, - vyIndex);
        }
        return schemeChange(xIndex, yIndex, vxIndex, vyIndex);
    }

    inline double limiter(double theta) {
        return std::max(0.0, std::min(1.0, theta));
    }

    inline double limitValueX(double gammaX, int xIndex, int yIndex, int vxIndex, int vyIndex) {
        double value, thetaNom;
        double thetaDenom = prev->getValue(xIndex + 1, yIndex, vxIndex, vyIndex) - prev->getValue(xIndex, yIndex, vxIndex, vyIndex);
        if (vxIndex > 0) {
            thetaNom = prev->getValue(xIndex, yIndex, vxIndex, vyIndex) - prev->getValue(xIndex, yIndex - 1, vxIndex, vyIndex);
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
        if (vyIndex > 0) {
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

    void exportDensity(std::ostream* stream) {
        for (int xIndex = 0; xIndex < curr->xIndexMax; xIndex ++) {
            for (int yIndex = 0; yIndex < curr->yIndexMax; yIndex ++) {
                (*stream)
                        << grid->getX(xIndex) << "\t"
                        << grid->getY(yIndex) << "\t"
                        << this->getDensity(xIndex, yIndex) << "\n";
            }
            (*stream) << std::endl;
        }
    }

    double getDensity(int xIndex, int yIndex) {
        double value = 0, vx, vy;
        for (int vyIndex = prev->vyIndexMin; vyIndex < prev->vyIndexMax; vyIndex ++) {
            for (int vxIndex = prev->vxIndexMin; vxIndex < prev->vxIndexMax; vxIndex ++) {
                value += curr->getValue(xIndex, yIndex, vxIndex, vyIndex);
            }
        }
        return value;
    }
};


#endif //CALC_EVOLUTION2D_H
