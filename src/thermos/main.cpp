//
// Created by geoolekom on 25.10.18.
//

#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include "Data.h"

void toFile(const double* data, const int nx, const double xStep, const char *filename) {
    std::ofstream file;
    file.open(filename);
    for (int xIndex = 0; xIndex < nx; xIndex ++) {
        file << (double) xIndex * xStep << "\t" << data[xIndex] << std::endl;
    }
    file.close();
}

void getVLimits(int size, int rank, int nvFull, int* nvStart, int* nvEnd) {
    int i = 0, nv = nvFull;
    int *diffs = new int[size]();
    while (nv --) {
        diffs[i % size] ++;
        i ++;
    }
    *nvStart = - nvFull / 2;
    for (i = 0; i < rank; i ++) {
        *nvStart += diffs[i];
    }
    *nvEnd = *nvStart + diffs[rank];
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Input parsing
    if (argc < 7) {
        std::cout << "Недостаточно аргументов.\n";
        return 1;
    }
    char *endptr;
    int nx = (int) strtol(argv[1], &endptr, 10), nv = (int) strtol(argv[2], &endptr, 10);
    double highT = strtod(argv[3], &endptr),
           lowT = strtod(argv[4], &endptr),
           eps = strtod(argv[5], &endptr);
    char *outputFile = argv[6];
    if (nv % 2 == 0) {
        nv --;
    }

    // Limits calculation
    int nvStart, nvEnd;
    getVLimits(size, rank, nv, &nvStart, &nvEnd);
    std::cout << nvStart << " " << nvEnd << std::endl;

    auto* data = new Data(rank, size, nx, nv, highT, lowT, eps, nvStart, nvEnd);
    data->setInitialValues();

    double startTime = MPI_Wtime();
    data->solve();
    double endTime = MPI_Wtime();
    std::cout << (endTime - startTime) * 1000 << " мс." << std::endl;

    if (rank == 0) {
        toFile(data->getTemperature(), data->getNx(), data->getXStep(), outputFile);
    }

    MPI_Finalize();
    return 0;
}

