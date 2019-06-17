#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <host_defines.h>
#include <cuda_runtime_api.h>

#include "utils/cuda.h"
#include "interfaces/Geometry.h"
#include "State3D.h"
#include "Grid3D.h"
#include "Evolution3D.h"
#include "IndexTankWithScreen2D.cu"
#include "Storage2D.h"
#include "DoduladCI.h"

void setInitialValues(State3D* state, Grid3D* grid, IndexTankWithScreen2D* geometry) {
    double denom = 0, value;
    doubleVector v;

    for (const auto& vIndex : state->getVelocityIterable()) {
        if (grid->inBounds(vIndex)) {
            v = grid->getV(vIndex);
            denom += exp(- (v * v) / 2);
        }
    }

    for (const auto& xIndex : state->getSpaceIterable()) {
        for (const auto& vIndex : state->getVelocityIterable()) {
            v = grid->getV(vIndex);
            value = exp(- (v * v) / 2) / denom;
            if (!geometry->isInTank(xIndex)) {
                value /= 1e8;
            }
            state->setValue(xIndex, vIndex, value);
        }
    }
}


__global__ void evolve(Evolution3D* e, int step) {
    int txIndex = threadIdx.x;
    int txStep = blockDim.x;
    int tyIndex = blockIdx.x * blockDim.y + threadIdx.y;
    int tyStep = gridDim.x * blockDim.y;
    e->makeStep(step, txIndex, tyIndex, txStep, tyStep);
}


int main(int argc, char* argv[]) {
    char dataDir[] = "data/06.05.19/9/flow";
    const int k = 1;
    const int step = 10 * k;

    int vRadius = 48;
    double vMax = 4.80;
    double tStep = 1e-1 / k, xStep = 1.0 / k, vStep = vMax / vRadius;

    int height = 25 * k, length = 100 * k;
    int wallY = 7 * k, screenY = 7 * k;
    int wallLeftX = 25 * k, wallRightX = wallLeftX + 2;
    int screenLeftX = 25 * k, screenRightX = screenLeftX + 2;

    printf("Выделение памяти.\n");

    IndexTankWithScreen2D* geometry;
    auto tempGeometry = IndexTankWithScreen2D(wallLeftX, wallRightX, wallY, height, screenLeftX, screenRightX, screenY, length);
    cudaCopy(&geometry, &tempGeometry);

    Grid3D* grid;
    auto tempGrid = Grid3D({xStep, xStep, xStep}, {vStep, vStep, vMax}, vMax);
    cudaCopy(&grid, &tempGrid);

//    State3D* state;
//    auto tempState = State3D({length, height, 1}, {-vRadius, -vRadius, -vRadius}, {vRadius, vRadius, vRadius});
//    cudaCopy(&state, &tempState);
//    state->cudaAllocate();

    State3D* state = new State3D({length, height, 1}, {-vRadius, -vRadius, -1}, {vRadius, vRadius, 1});
    state->allocate();

    setInitialValues(state, grid, geometry);

    DoduladCI* ci;
    auto tempCi = DoduladCI(tStep, vStep, vRadius, state);
    cudaCopy(&ci, &tempCi);

    Evolution3D* evolution;
    auto tempEvolution = Evolution3D(tStep, state, grid, geometry, ci);
    cudaCopy(&evolution, &tempEvolution);

    auto storage = new Storage2D(state, grid);
    std::ofstream file;
    char filename[100];
    const auto startTime = std::time(nullptr);

    size_t available, total;
    cudaMemGetInfo(&available, &total);
    printf("Занято видеопамяти: %zu Мб / %zu Мб\n", (total - available) / 1024 / 1024, total / 1024 / 1024);
    printf("Начинаем обсчет.\n");
    for (int i = 0; i < 3000 * k; i++) {
        ci->generateGrid();
        evolve<<<dim3(5, 1, 1), dim3(100, 5, 1)>>>(evolution, i);
        auto ret = cudaDeviceSynchronize();
        ci->finalizeGrid();
        if (ret != 0) {
            std::cout << "Ошибка: " << cudaGetErrorString(ret) << std::endl;
            break;
        }
        evolution->swap();

        auto duration = std::time(nullptr) - startTime;
        printf("%02zu:%02zu\tШаг %d\n", duration / 60, duration % 60, i);

        if (i % step == 0) {
            evolution->exportToHost();
            std::cout << "Запись в файлы...\n";

            sprintf(filename, "%s/data_%03d.out", dataDir, i);
            file.open(filename);
            storage->exportAll(&file);
            file.close();

            sprintf(filename, "%s/radius_%03d.out", dataDir, i);
            file.open(filename);
            storage->exportRadius(&file);
            file.close();

            sprintf(filename, "%s/mach_0_%03d.out", dataDir, i);
            file.open(filename);
            storage->exportMachNumber(&file, 0);
            file.close();
//
//            sprintf(filename, "data/flow/temperature_xx_%03d.out", i);
//            file.open(filename);
//            storage->exportTemperatureTensor(&file, {0, 0});
//            file.close();
//
//            sprintf(filename, "data/flow/temperature_yy_%03d.out", i);
//            file.open(filename);
//            storage->exportTemperatureTensor(&file, {1, 1});
//            file.close();

            sprintf(filename, "%s/flow.out", dataDir);
            file.open(filename, std::ofstream::app);
            storage->exportFlowX(&file, i, screenRightX, screenY);
            file.close();

            sprintf(filename, "%s/function_screen_0_%03d.out", dataDir, i);
            file.open(filename);
            storage->exportFunction(&file, screenRightX, 0);
            file.close();

            sprintf(filename, "%s/function_screen_10_0_%03d.out", dataDir, i);
            file.open(filename);
            storage->exportFunction(&file, screenRightX + 10 * k, 0);
            file.close();
        }
    }
    cudaFree(evolution);
    state->release();
    delete state;
    cudaFree(grid);
    cudaFree(geometry);
    delete storage;
    return 0;
}
