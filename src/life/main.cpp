//
// Created by geoolekom on 01.10.18.
//

#include <iostream>
#include <mpi.h>
#include "LifeData.h"
#include "DistributedLifeData.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    if (argc != 2) {
        std::cout << "No input file detected.\n";
        return 0;
    }
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double startTime = MPI_Wtime();

    DistributedLifeData* l = new DistributedLifeData(argv[1], size, rank);
    l->evolve();
    double endTime = MPI_Wtime();
    std::cout << (endTime - startTime) * 1000 << " ms." << std::endl;

    MPI_Finalize();
    delete l;
    return 0;
}