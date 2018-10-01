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
    DistributedLifeData* l = new DistributedLifeData(argv[1]);
    l->evolve();
    MPI_Finalize();
    return 0;
}