#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <ctime>

#include "State2D.h"
#include "Grid2D.h"
#include "Tank2D.h"
#include "TankWithScreen2D.h"
#include "IndexTankWithScreen2D.h"
#include "IndexTankFull2D.h"
#include "Evolution2D.h"
#include "Storage2D.h"
#include "DoduladCI.h"


void setInitialValues(State2D* state, Grid2D* grid, IndexTankWithScreen2D* geometry) {
    double denom = 0, value;
    doubleVector v;

    for (const auto& vIndex : state->getVelocityIterable()) {
        v = grid->getV(vIndex);
        if (grid->inBounds(v)) {
            denom += exp(- (v * v) / 2);
        }
    }

    for (const auto& xIndex : state->getSpaceIterable()) {
        for (const auto& vIndex : state->getVelocityIterable()) {
            v = grid->getV(vIndex);
            if (grid->inBounds(v)) {
                value = exp(- (v * v) / 2) / denom;
                if (!geometry->isInTank(xIndex)) {
                    value /= 1e8;
                }
            } else {
                value = -1;
            }
            state->setValue(xIndex, vIndex, value);
        }
    }
}


int main(int argc, char* argv[]) {
    const int k = 2;
    const int step = 10 * k;
    std::cout << "Начинаем обсчет.\n";

    int wallY = 5, screenY = 5;
    int wallLeftX = 25 * k, wallRightX = wallLeftX + 2;
    int screenLeftX = 25 * k, screenRightX = screenLeftX + 2;

    auto state = new State2D(100 * k + 1, 25 * k + 1, -48, 48, -48, 48);
    auto geometry = new IndexTankWithScreen2D(wallLeftX, wallRightX, wallY, 25 * k + 1, screenLeftX, screenRightX, screenY, 100 * k + 1);
    auto grid = new Grid2D(1.0 / k, 1.0 / k, 0.1, 0.1, 4.85);
    setInitialValues(state, grid, geometry);

    double tStep = 0.1 / k;
    auto ci = new DoduladCI(tStep, grid->vxStep, state);
    auto evolution = new Evolution2D(tStep, &state, grid, geometry, ci);
    auto storage = new Storage2D(state, grid);
    std::ofstream file;
    char filename[40], formatString[40];
    const auto startTime = std::time(nullptr);

    for (int i = 0; i < 6000; i++) {
        evolution->evolve(i);
        auto duration = std::time(nullptr) - startTime;
        sprintf(formatString, "%02d:%02d\tШаг %d\n", (int) duration / 60, (int) duration % 60, i);
        std::cout << formatString;

        if (i % step == 0) {
            std::cout << "Запись в файлы...\n";

            sprintf(filename, "data/flow/data_%03d.out", i);
            file.open(filename);
            storage->exportAll(&file);
            file.close();

//            sprintf(filename, "data/flow/density_%03d.out", i);
//            file.open(filename);
//            storage->exportDensity(&file);
//            file.close();

            sprintf(filename, "data/flow/radius_%03d.out", i);
            file.open(filename);
            storage->exportRadius(&file);
            file.close();

            sprintf(filename, "data/flow/mach_0_%03d.out", i);
            file.open(filename);
            storage->exportMachNumber(&file, 0);
            file.close();

//            sprintf(filename, "data/flow/temperature_%03d.out", i);
//            file.open(filename);
//            storage->exportTemperature(&file);
//            file.close();

            sprintf(filename, "data/flow/temperature_xx_%03d.out", i);
            file.open(filename);
            storage->exportTemperatureTensor(&file, {0, 0});
            file.close();

            sprintf(filename, "data/flow/temperature_yy_%03d.out", i);
            file.open(filename);
            storage->exportTemperatureTensor(&file, {1, 1});
            file.close();

            sprintf(filename, "data/flow/flow.out");
            file.open(filename, std::ofstream::app);
            storage->exportFlowX(&file, i, screenRightX, screenY);
            file.close();

//            sprintf(filename, "data/flow/velocity_%03d.out", i);
//            file.open(filename);
//            storage->exportVelocity(&file);
//            file.close();

            sprintf(filename, "data/flow/function_screen_0_%03d.out", i);
            file.open(filename);
            storage->exportFunction(&file, screenRightX, 0);
            file.close();

            sprintf(filename, "data/flow/function_screen_10_0_%03d.out", i);
            file.open(filename);
            storage->exportFunction(&file, screenRightX + 10 * k, 0);
            file.close();
        }
    }
    delete evolution;
    delete state;
    delete grid;
    delete geometry;
    delete storage;
    return 0;
}
