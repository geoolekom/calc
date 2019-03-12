#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>

#include "State2D.h"
#include "Grid2D.h"
#include "Tank2D.h"
#include "Evolution2D.h"
#include "Storage2D.h"
#include "DoduladCI.h"


void setInitialValues(State2D* state, Grid2D* grid, Tank2D* geometry) {
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
                if (!geometry->isInTank(grid->getX(xIndex))) {
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

    auto state = new State2D(100 * k + 1, 25 * k + 1, -10, 10, -10, 10);
    auto geometry = new Tank2D(5, 5.2, 0.6, 5, 5.6, 5.8, 0.6, 20);
    auto grid = new Grid2D(0.2 / k, 0.2 / k, 0.5, 0.5, 5);
    setInitialValues(state, grid, geometry);

    double tStep = 2e-2 / k;
    auto ci = new DoduladCI(tStep, grid->vxStep, state);
    auto evolution = new Evolution2D(tStep, &state, grid, geometry, ci);
    auto storage = new Storage2D(state, grid);
    std::ofstream file;
    char filename[40];

    for (int i = 0; i < 2000; i++) {
        evolution->evolve(i);
        std::cout << "Шаг " << i << "\n";

        if (i % step == 0) {
            std::cout << "Запись в файлы...\n";
            sprintf(filename, "data/flow/density_%03d.out", i);
            file.open(filename);
            storage->exportDensity(&file);
            file.close();

            sprintf(filename, "data/flow/temperature_%03d.out", i);
            file.open(filename);
            storage->exportTemperature(&file);
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
            storage->exportFunction(&file, 30 * k, 3);
            file.close();

            sprintf(filename, "data/flow/function_50_10_%03d.out", i);
            file.open(filename);
            storage->exportFunction(&file, 50 * k, 5);
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
