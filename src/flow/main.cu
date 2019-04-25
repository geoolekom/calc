#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <host_defines.h>
#include <cuda_runtime_api.h>

#include "State2D.cu"
#include "Grid2D.cu"
#include "Tank2D.h"
#include "TankWithScreen2D.h"
#include "IndexTankWithScreen2D.cu"
#include "IndexTankFull2D.h"
#include "Evolution2D.cu"
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


__global__ void kernel(Evolution2D* e, int step) {
    int txIndex = threadIdx.x;
    int txStep = blockDim.x;
    int tyIndex = blockIdx.x;
    int tyStep = gridDim.x;
    e->makeStep(step, txIndex, txStep, tyIndex, tyStep);
}


int main(int argc, char* argv[]) {
    const int k = 1;
    const int step = 10 * k;

    double tStep = 0.1 / k;
    int height = 25 * k + 1, length = 100 * k + 1;
    int wallY = 5, screenY = 5;
    int wallLeftX = 25 * k, wallRightX = wallLeftX + 2;
    int screenLeftX = 25 * k, screenRightX = screenLeftX + 2;

    auto tempGeometry = IndexTankWithScreen2D(wallLeftX, wallRightX, wallY, height, screenLeftX, screenRightX, screenY, length);
    IndexTankWithScreen2D* geometry;
    cudaMallocManaged((void**) &geometry, sizeof(IndexTankWithScreen2D));
    cudaMemcpy(geometry, &tempGeometry, sizeof(IndexTankWithScreen2D), cudaMemcpyHostToDevice);

    auto tempGrid = Grid2D(1.0 / k, 1.0 / k, 0.1, 0.1, 4.85);
    Grid2D* grid;
    cudaMallocManaged((void**) &grid, sizeof(Grid2D));
    cudaMemcpy(grid, &tempGrid, sizeof(Grid2D), cudaMemcpyHostToDevice);

    State2D* state = new State2D(length, height, -48, 48, -48, 48);
    state->allocate();
    setInitialValues(state, grid, geometry);

//    double* initialData;  // = new double[state->getSize()]();
//    cudaMallocManaged((void**) &initialData, sizeof(double) * state->getSize());
//    state->setData(initialData);
//    setInitialValues(state, grid, geometry);

    auto tempCi = DoduladCI(tStep, grid->vxStep, state);
    DoduladCI* ci;
    cudaMallocManaged((void**) &ci, sizeof(DoduladCI));
    cudaMemcpy(ci, &tempCi, sizeof(DoduladCI), cudaMemcpyHostToDevice);

    auto tempEvolution = Evolution2D(tStep, state, grid, geometry, ci);
    Evolution2D* evolution;
    cudaMallocManaged((void**) &evolution, sizeof(Evolution2D));
    cudaMemcpy(evolution, &tempEvolution, sizeof(Evolution2D), cudaMemcpyHostToDevice);

    auto storage = new Storage2D(state, grid);
    std::ofstream file;
    char filename[40], formatString[40];
    const auto startTime = std::time(nullptr);

    size_t available, total;
    cudaMemGetInfo(&available, &total);
    std::cout << "Доступно видеопамяти: " << available << "/" << total << std::endl;
    std::cout << "Начинаем обсчет.\n";
    for (int i = 0; i < 3000 * k; i++) {
        kernel<<<8,128 * k>>>(evolution, i);
        auto ret = cudaDeviceSynchronize();
        if (ret != 0) {
            std::cout << "Код ошибки: " << ret << std::endl;
            break;
        }
        evolution->swap();

        auto duration = std::time(nullptr) - startTime;
        sprintf(formatString, "%02d:%02d\tШаг %d\n", (int) duration / 60, (int) duration % 60, i);
        std::cout << formatString;

        if (i % step == 0) {
            evolution->exportToHost();
            std::cout << "Запись в файлы...\n";

            sprintf(filename, "data/flow_2/data_%03d.out", i);
            file.open(filename);
            storage->exportAll(&file);
            file.close();

//            sprintf(filename, "data/flow/radius_%03d.out", i);
//            file.open(filename);
//            storage->exportRadius(&file);
//            file.close();
//
//            sprintf(filename, "data/flow/mach_0_%03d.out", i);
//            file.open(filename);
//            storage->exportMachNumber(&file, 0);
//            file.close();
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
//
//            sprintf(filename, "data/flow/flow.out");
//            file.open(filename, std::ofstream::app);
//            storage->exportFlowX(&file, i, screenRightX, screenY);
//            file.close();
//
//            sprintf(filename, "data/flow/function_screen_0_%03d.out", i);
//            file.open(filename);
//            storage->exportFunction(&file, screenRightX, 0);
//            file.close();
//
//            sprintf(filename, "data/flow/function_screen_10_0_%03d.out", i);
//            file.open(filename);
//            storage->exportFunction(&file, screenRightX + 10 * k, 0);
//            file.close();
        }
    }
    cudaFree(evolution);
//    cudaFree(initialData);
//    delete[] initialData;
    state->free();
    delete state;
    cudaFree(grid);
    cudaFree(geometry);
    delete storage;
    return 0;
}
