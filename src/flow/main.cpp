#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "State2D.h"
#include "Grid2D.h"
#include "Tank2D.h"
#include "Evolution2D.h"
#include "Storage2D.h"
#include "VagapovaCI.h"


void setInitialValues(State2D* state, Grid2D* grid, Tank2D* geometry) {
    double denom = 0, vx, vy;
    for (int vyIndex = state->vyIndexMin; vyIndex < state->vyIndexMax; vyIndex ++) {
        for (int vxIndex = state->vxIndexMin; vxIndex < state->vxIndexMax; vxIndex ++) {
            vx = grid->getVx(vxIndex);
            vy = grid->getVy(vyIndex);
            denom += exp(- (vx * vx + vy * vy) / 2);
        }
    }
    for (int yIndex = 0; yIndex < state->yIndexMax; yIndex ++) {
        for (int xIndex = 0; xIndex < state->xIndexMax; xIndex ++) {
            for (int vyIndex = state->vyIndexMin; vyIndex < state->vyIndexMax; vyIndex ++) {
                for (int vxIndex = state->vxIndexMin; vxIndex < state->vxIndexMax; vxIndex ++) {
                    vx = grid->getVx(vxIndex);
                    vy = grid->getVy(vyIndex);
                    double value = exp(- (vx * vx + vy * vy) / 2) / denom;
                    if (xIndex >= geometry->rightWallLeftX) {
                        value /= 1e6;
                    }
                    state->setValue(xIndex, yIndex, vxIndex, vyIndex, value);
                }
            }
        }
    }
}


int main(int argc, char* argv[]) {
    auto state = new State2D(50, 25, -20, 21, -20, 21);
    auto geometry = new Tank2D(25, 26, 3, 25, 30, 31, 3, 50);
    auto grid = new Grid2D(0.2, 0.2, 0.25, 0.25);
    setInitialValues(state, grid, geometry);

    double tStep = 2e-2;
    auto ci = new VagapovaCI(tStep, grid->vxStep, state);
    auto evolution = new Evolution2D(tStep, &state, grid, geometry, ci);
    auto storage = new Storage2D(state, grid);
    std::ofstream file;
    char filename[30];
    for (int i = 0; i < 200; i++) {
        evolution->evolve(i);
        sprintf(filename, "data/flow/density_%03d.out", i);
        file.open(filename);
        storage->exportDensity(&file);
        std::cout << "Шаг " << i << "\n";
        file.close();
    }
    delete evolution;
    delete state;
    delete grid;
    delete geometry;
    delete storage;
    return 0;
}
