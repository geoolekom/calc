#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>

#include "DoduladCI.h"
#include "Evolution3D.h"
#include "Grid3D.h"
#include "State3D.h"
#include "Storage2D.hpp"
#include "interfaces/Geometry.h"
#include "utils/cuda.h"

std::function<void(int)> shutdownHandler;
void signalHandler(int signal) { shutdownHandler(signal); }

void setInitialValues(State3D *state, Grid3D *grid, GeometryInterface *geometry) {
    double denom = 0;
    double value;
    doubleVector v;

    for (const auto &vIndex : state->getVelocityIterable()) {
        if (grid->inBounds(vIndex)) {
            v = grid->getV(vIndex);
            denom += exp(-(v * v) / 2);
        }
    }

    for (const auto &xIndex : state->getSpaceIterable()) {
        for (const auto &vIndex : state->getVelocityIterable()) {
            v = grid->getV(vIndex);
            value = exp(-(v * v) / 2) / denom;
            if (!geometry->isInTank(xIndex)) {
                value /= 1e8;
            }
            state->setValue(xIndex, vIndex, value);
        }
    }
}

__global__ void evolve(Evolution3D *e, int step) {
    int txIndex = (int)threadIdx.x;                             // NOLINT(readability-static-accessed-through-instance)
    int txStep = (int)blockDim.x;                               // NOLINT(readability-static-accessed-through-instance)
    int tyIndex = (int)(blockIdx.x * blockDim.y + threadIdx.y); // NOLINT(readability-static-accessed-through-instance)
    int tyStep = (int)(gridDim.x * blockDim.y);                 // NOLINT(readability-static-accessed-through-instance)
    int tzIndex = (int)(blockIdx.y * blockDim.z + threadIdx.z); // NOLINT(readability-static-accessed-through-instance)
    int tzStep = (int)(gridDim.y * blockDim.z);                 // NOLINT(readability-static-accessed-through-instance)
    e->makeStep(step, txIndex, tyIndex, tzIndex, txStep, tyStep, tzStep);
}

int main(int argc, char *argv[]) {
    char dataDir[] = "data/flow";
    const int k = 1;
    const int step = 10 * k;
    const int epochCount = 500 * k;

    int vRadius = 5;
    // Занимаемая память пропорциональна k^3 * vRadius^3
    double vMax = 4.80;
    double tStep = 1e-1 / k;
    double xStep = 1.0 / k;
    double vStep = vMax / vRadius;

    int height = 25 * k;
    int length = 100 * k;
    int width = 25 * k;
    int wallLeftX = 25 * k;
    int wallRightX = wallLeftX + 2;
    int holeCenterY = 6 * k;
    int holeCenterZ = 6 * k;
    int holeRadius = 4 * k;

    printf("Выделение памяти.\n");

    GeometryInterface *geometry;
    // IndexTankWithScreen2D *geometry;
    auto tempGeometry =
        GeometryInterface(holeCenterY, holeCenterZ, holeRadius, wallLeftX, wallRightX, height, width, length);
    cudaCopy(&geometry, &tempGeometry);

    Grid3D *grid;
    auto tempGrid = Grid3D({xStep, xStep, xStep}, {vStep, vStep, vStep}, vMax);
    cudaCopy(&grid, &tempGrid);

    //    State3D* state;
    //    auto tempState = State3D({length, height, 1}, {-vRadius, -vRadius, -vRadius}, {vRadius, vRadius, vRadius});
    //    cudaCopy(&state, &tempState);
    //    state->cudaAllocate();

    auto *state = new State3D({length, height, width}, {-vRadius, -vRadius, -vRadius}, {vRadius, vRadius, vRadius});
    state->allocate();

    // setInitialValues(state, grid, geometry);

    auto *storage = new Storage2D(state, grid);
    std::ofstream file;
    char filename[100];

    // Импортируем данные из файла
    std::ifstream inputFile;
    sprintf(filename, "%s/state.in", dataDir);
    inputFile.open(filename);
    storage->importState(&inputFile);

    shutdownHandler = [storage, dataDir](int signal) {
        std::ofstream file;
        char filename[100];
        std::cout << "Сохраняем состояние...\n";
        sprintf(filename, "%s/state.out", dataDir);
        file.open(filename);
        storage->exportState(&file);
        file.close();
        std::cout << "Завершаем приложение.\n";
        exit(signal);
    };
    std::signal(SIGINT, signalHandler);

    const auto startTime = std::time(nullptr);

    // Сохраняем параметры запуска
    char paramsString[200];
    tempGeometry.serializeParams(paramsString);
    sprintf(filename, "%s/params.txt", dataDir);
    file.open(filename);
    file << paramsString;
    file.close();

    DoduladCI *ci;
    auto tempCi = DoduladCI(tStep, vStep, vRadius, state);
    cudaCopy(&ci, &tempCi);

    Evolution3D *evolution;
    auto tempEvolution = Evolution3D(tStep, state, grid, geometry, ci);
    cudaCopy(&evolution, &tempEvolution);

    size_t available;
    size_t total;
    cudaMemGetInfo(&available, &total);
    printf("Занято видеопамяти: %zu Мб / %zu Мб\n", (total - available) / 1024 / 1024, total / 1024 / 1024);
    printf("Начинаем обсчет.\n");
    for (int i = 0; i < epochCount; i++) {
        for (int j = 0; j < 3; j++) {
            ci->generateGrid();
            evolve<<<dim3(32, 32, 1), dim3(128, 1, 1)>>>(evolution, j);
            auto errorCode = cudaGetLastError();
            if (errorCode != 0) {
                std::cout << "Ошибка cudaGetLastError: " << cudaGetErrorString(errorCode) << std::endl;
                break;
            }
            ci->finalizeGrid();
            errorCode = cudaDeviceSynchronize();
            if (errorCode != 0) {
                std::cout << "Ошибка cudaDeviceSynchronize: " << cudaGetErrorString(errorCode) << std::endl;
                break;
            }
            evolution->swap();
        }

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

            // sprintf(filename, "data/flow/temperature_xx_%03d.out", i);
            // file.open(filename);
            // storage->exportTemperatureTensor(&file, {0, 0});
            // file.close();
            //
            // sprintf(filename, "data/flow/temperature_yy_%03d.out", i);
            // file.open(filename);
            // storage->exportTemperatureTensor(&file, {1, 1});
            // file.close();
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
