//
// Created by geoolekom on 05.10.18.
//

#include <iostream>
#include <fstream>
#include <mpi.h>
#include "Worker.h"
#include "Master.h"


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    if (argc != 3) {
        std::cout << "Input file name and output directory are not specified or ambiguous.\n";
        return 0;
    }
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int root = size - 1;
    const char* inputFileName = argv[1];
    const char* outputDir = argv[2];
    double startTime = MPI_Wtime();

    int stepCount, saveFrequency, nx, ny, *array = nullptr;

    std::ifstream file;
    file.open(inputFileName);

    file >> stepCount >> saveFrequency >> nx >> ny;
    array = new int[nx * ny];
    int i, j;
    while (file >> i >> j) {
        array[i + nx * j] = 1;
    }
    file.close();

    Data* initialData = new Data(nx, ny, array);
    delete[] array;

    if (rank == root) {
        char path[100];
        auto* m = new Master(size - 1, initialData);
        for (int i = 0; i < stepCount / saveFrequency; i++) {
            m->wait(i);
            sprintf(path, "%s/life_%06d.vtk", outputDir, i * saveFrequency);
            m->getData()->toFile(path);
        }
        delete m;
    } else {
        auto* w = new Worker(rank, size - 1, initialData);
        for (int i = 0; i < stepCount; i ++) {
            if (i % saveFrequency == 0) {
                w->sendData(i / saveFrequency);
            }
            w->makeStep(i);
        }
        delete w;
    }

    double endTime = MPI_Wtime();
    std::cout << (endTime - startTime) * 1000 << " ms." << std::endl;

    MPI_Finalize();
    return 0;
}
