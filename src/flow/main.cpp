#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "State2D.h"
#include "Grid2D.h"
#include "Tank2D.h"
#include "Evolution2D.h"


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
                    if (xIndex >= grid->getXIndex(geometry->xWallStart)) {
                        value /= 10e6;
                    }
                    state->setValue(xIndex, yIndex, vxIndex, vyIndex, value);
                }
            }
        }
    }
}


int main(int argc, char* argv[]) {
    auto* geometry = new Tank2D(5, 5.5, 2);
    auto* grid = new Grid2D(0.2, 0.2, 0.25, 0.25);
    auto* state = new State2D(50, 25, -20, 20, -20, 20);
    setInitialValues(state, grid, geometry);

    double tStep = 1e-2;
    auto* evolution = new Evolution2D(tStep, &state, grid, geometry);
    std::ofstream file;
    char filename[30];
    for (int i = 0; i < 100; i++) {
        evolution->evolve(i);
        sprintf(filename, "data/flow/density_%03d.out", i);
        file.open(filename);
        evolution->exportDensity(&file);
        std::cout << "Шаг " << i << "\n";
        file.close();
    }

    delete evolution;
    delete state;
    delete grid;
    delete geometry;
    return 0;
}
