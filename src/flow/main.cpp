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
                    value /= 1e6;
                }
            } else {
                value = -1;
            }
            state->setValue(xIndex, vIndex, value);
        }
    }
}


int main(int argc, char* argv[]) {
    const int k = 1;
    const int step = 10;
    std::cout << "Начинаем обсчет.\n";

    auto state = new State2D(100 * k + 1, 25 * k + 1, -25, 25, -25, 25);
    auto geometry = new IndexTankWithScreen2D(25 * k, 25 * k + 2, 4, 25 * k + 1, 40 * k, 40 * k + 2, 4, 100 * k + 1);
    auto grid = new Grid2D(1.0 / k, 1.0 / k, 0.2, 0.2, 4.8);
    setInitialValues(state, grid, geometry);

    double tStep = 1e-1 / k;
    auto ci = new CollisionIntegral<State2D>();
    auto evolution = new Evolution2D(tStep, &state, grid, geometry, ci);
    auto storage = new Storage2D(state, grid);
    std::ofstream file;
    char filename[40], formatString[40];
    const auto startTime = std::time(nullptr);

    for (int i = 0; i < 2000; i++) {
        evolution->evolve(i);
        auto duration = std::time(nullptr) - startTime;
        sprintf(formatString, "%02d:%02d\tШаг %d\n", (int) duration / 60, (int) duration % 60, i);
        std::cout << formatString;

        if (i % step == 0) {
            std::cout << "Запись в файлы...\n";
            sprintf(filename, "data/flow/density_%03d.out", i);
            file.open(filename);
            storage->exportDensity(&file);
            file.close();

            sprintf(filename, "data/flow/radius_%03d.out", i);
            file.open(filename);
            storage->exportRadius(&file);
            file.close();

            sprintf(filename, "data/flow/mach_0_%03d.out", i);
            file.open(filename);
            storage->exportMachNumber(&file, 0);
            file.close();

            sprintf(filename, "data/flow/mach_4_%03d.out", i);
            file.open(filename);
            storage->exportMachNumber(&file, 4);
            file.close();

            sprintf(filename, "data/flow/temperature_%03d.out", i);
            file.open(filename);
            storage->exportTemperature(&file);
            file.close();

            sprintf(filename, "data/flow/temperature_xx_%03d.out", i);
            file.open(filename);
            storage->exportTemperatureTensor(&file, {0, 0});
            file.close();

            sprintf(filename, "data/flow/temperature_yy_%03d.out", i);
            file.open(filename);
            storage->exportTemperatureTensor(&file, {1, 1});
            file.close();

            sprintf(filename, "data/flow/flow_30.out");
            file.open(filename, std::ofstream::app);
            storage->exportFlowX(&file, i, 30 * k, 3 * k);
            file.close();

            sprintf(filename, "data/flow/flow_35.out");
            file.open(filename, std::ofstream::app);
            storage->exportFlowX(&file, i, 35 * k, 3 * k);
            file.close();

            sprintf(filename, "data/flow/flow_50.out");
            file.open(filename, std::ofstream::app);
            storage->exportFlowX(&file, i, 50 * k, 3 * k);
            file.close();

            sprintf(filename, "data/flow/function_28_10_%03d.out", i);
            file.open(filename);
            storage->exportFunction(&file, 28 * k, 10 * k);
            file.close();

            sprintf(filename, "data/flow/function_30_3_%03d.out", i);
            file.open(filename);
            storage->exportFunction(&file, 30 * k, 3 * k);
            file.close();

            sprintf(filename, "data/flow/function_30_0_%03d.out", i);
            file.open(filename);
            storage->exportFunction(&file, 30 * k, 0);
            file.close();

            sprintf(filename, "data/flow/function_32_0_%03d.out", i);
            file.open(filename);
            storage->exportFunction(&file, 32 * k, 0);
            file.close();

            sprintf(filename, "data/flow/function_40_0_%03d.out", i);
            file.open(filename);
            storage->exportFunction(&file, 40 * k, 0);
            file.close();

            sprintf(filename, "data/flow/function_50_5_%03d.out", i);
            file.open(filename);
            storage->exportFunction(&file, 50 * k, 5 * k);
            file.close();

            sprintf(filename, "data/flow/velocity_%03d.out", i);
            file.open(filename);
            storage->exportVelocity(&file);
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
