//
// Created by geoolekom on 21.02.19.
//

#include <vector>
#include <cmath>
#include <fstream>
#include "interfaces/Space.h"
#include "interfaces/Evolution.h"
#include "interfaces/Geometry.h"
#include "interfaces/Grid.h"
#include "interfaces/Storage.h"


class Evolution2D : public Evolution<2> {
public:
    using Evolution<2>::Evolution;

    double calculateDistributionFunction(const vectorType& xIndex, const vectorType& vIndex) override {
        auto v = grid->getVelocity(vIndex - vectorType({5, 5}));
        double gammaX = v[0] * tStep / 0.2;
        double gammaY = v[1] * tStep / 0.2;
        return prev->getValue(xIndex, vIndex)
               - gammaX * (limitValueX(gammaX, xIndex, vIndex) - limitValueX(gammaX, xIndex - vectorType({1, 0}), vIndex))
               - gammaY * (limitValueY(gammaY, xIndex, vIndex) - limitValueY(gammaY, xIndex - vectorType({0, 1}), vIndex));
    }

    inline double limiter(double theta) {
        return std::max(0.0, std::min(1.0, theta));
    }

    inline double limitValueX(double gammaX, const vectorType& xIndex, const vectorType& vIndex) {
        double value, thetaNom;
        vectorType shift1 = {{1, 0}}, shift2 = {{2, 0}};
        double thetaDenom = prev->getValue(xIndex + shift1, vIndex) - prev->getValue(xIndex, vIndex);
        if (vIndex[0] > 5) {
            thetaNom = prev->getValue(xIndex, vIndex) - prev->getValue(xIndex - shift1, vIndex);
            value = prev->getValue(xIndex, vIndex);
            return value + (1 - gammaX) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
        } else {
            thetaNom = prev->getValue(xIndex + shift2, vIndex) - prev->getValue(xIndex + shift1, vIndex);
            value = prev->getValue(xIndex + shift1, vIndex);
            return value - (1 + gammaX) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
        }
    }

    inline double limitValueY(double gammaY, const vectorType& xIndex, const vectorType& vIndex) {
        double value, thetaNom;
        vectorType shift1 = {{0, 1}}, shift2 = {{0, 2}};
        double thetaDenom = prev->getValue(xIndex + shift1, vIndex) - prev->getValue(xIndex, vIndex);
        if (vIndex[1] > 5) {
            thetaNom = prev->getValue(xIndex, vIndex) - prev->getValue(xIndex - shift1, vIndex);
            value = prev->getValue(xIndex, vIndex);
            return value + (1 - gammaY) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
        } else {
            thetaNom = prev->getValue(xIndex + shift2, vIndex) - prev->getValue(xIndex + shift1, vIndex);
            value = prev->getValue(xIndex + shift1, vIndex);
            return value - (1 + gammaY) * limiter(thetaNom / thetaDenom) * thetaDenom / 2.0;
        }
    }
};


class Tank2D : public Geometry<2> {
public:
    int rightWallLeftX, rightWallRightX, rightWallY;  // Правая стенка
    int topWallY;  // Верхняя стенка
    int screenLeftX, screenRightX, screenY;  // Экран
    int tankRightX;  // Правая граница области счета
public:
    Tank2D(int rightWallLeftX, int rightWallRightX, int rightWallY, int topWallY, int screenLeftX,
           int screenRightX, int screenY, int tankRightX) :
            rightWallLeftX (rightWallLeftX), rightWallRightX (rightWallRightX), rightWallY (rightWallY),
            topWallY (topWallY), screenLeftX (screenLeftX), screenRightX (screenRightX), screenY (screenY),
            tankRightX (tankRightX) {};

    bool isDiffuseReflection(const Vector& xIndex, const Vector& vIndex) override {
        if (xIndex[0] == 0 ||
            (xIndex[0] == rightWallRightX && xIndex[1] >= rightWallY) ||
            (xIndex[0] == screenRightX && xIndex[1] >= screenY)) {
            // летящие вправо имеют рассеянное распределение
            return vIndex[0] > 0;
        } else if ((xIndex[0] == rightWallLeftX && xIndex[1] >= rightWallY) ||
                   (xIndex[0] == screenLeftX && xIndex[1] >= screenY)) {
            // влево
            return vIndex[0] < 0;
        } else if (xIndex[1] == topWallY - 1 && xIndex[0] <= rightWallLeftX) {
            // вниз
            return vIndex[1] < 0;
        } else {
            return false;
        }
    }

    bool isMirrorReflection(const Vector& xIndex, const Vector& vIndex) override {
        return xIndex[1] == 0 && vIndex[1] > 0;
    }

    bool isBorderReached(const Vector& xIndex) override {
        return (xIndex[0] > rightWallRightX && xIndex[1] == topWallY - 1) || (xIndex[0] == tankRightX - 1);
    }

    Matrix getMirrorNormal(const Vector& xIndex) override {
        return {{
            {{1, 0}},
            {{0, -1}}
        }};
    }

    Vector getDiffusionMask(const Vector& xIndex) override {
        if (xIndex[0] == 0 ||
            (xIndex[0] == rightWallRightX && xIndex[1] >= rightWallY) ||
            (xIndex[0] == screenRightX && xIndex[1] >= screenY)) {
            // летящие вправо имеют рассеянное распределение
            return {{ 1, 0 }};
        } else if ((xIndex[0] == rightWallLeftX && xIndex[1] >= rightWallY) ||
                   (xIndex[0] == screenLeftX && xIndex[1] >= screenY)) {
            // влево
            return {{ -1, 0 }};
        } else if (xIndex[1] == topWallY - 1 && xIndex[0] <= rightWallLeftX) {
            // вниз
            return {{ 0, -1 }};
        } else {
            return {{ 0, 0 }};
        }
    }
};


class Grid2D : public Grid<2> {
public:
    double vStep, xStep;
public:
    Grid2D(double vStep, double xStep) : vStep(vStep), xStep(xStep) {};

    RealVector getVelocity(const IndexVector& vIndex) override {
        return {vStep * vIndex[0], vStep * vIndex[1]};
    };
    RealVector getCoordinates(const IndexVector& xIndex) override {
        return {xStep * xIndex[0], xStep * xIndex[1]};
    };
};

void setInitialValues(Space<2>* space, Grid2D* grid, Tank2D* geometry) {
    double denom = 0;
    for (const auto& vIndex : space->velocityIterable()) {
        auto v = grid->getVelocity(vIndex);
        denom += exp(- (v * v) / 2);
    }
    for (const auto& xIndex : space->spaceIterable()) {
        for (const auto& vIndex : space->velocityIterable()) {
            auto v = grid->getVelocity(vIndex);
            double value = exp( - (v * v) / 2) / denom;
            if (xIndex[0] > geometry->rightWallLeftX) {
                value /= 1e6;
            }
            space->setValue(xIndex, vIndex, value);
        }
    }
}


int main(int argc, char* argv[]) {
    auto space = new Space<2>({50, 25}, {11, 11});
    auto geometry = new Tank2D(25, 26, 3, 25, 30, 31, 3, 50);
    auto grid = new Grid2D(0.5, 0.2);
    setInitialValues(space, grid, geometry);

    double tStep = 1e-2;
    auto evolution = new Evolution2D(tStep, &space, geometry, grid);
    auto storage = new Storage<2>(space, grid);

    std::ofstream file;
    char filename[30];
    for (int i = 0; i < 200; i++) {
        evolution->evolve(1);
        sprintf(filename, "data/blow/density_%03d.out", i);
        file.open(filename);
        storage->exportDensity(&file);
        std::cout << "Шаг " << i + 1 << "\n";
        file.close();
    }

    delete storage;
    delete evolution;
    delete grid;
    delete geometry;
    delete space;
    return 0;
}