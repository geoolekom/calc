//
// Created by geoolekom on 01.10.18.
//

#include <iostream>
#include <mpi.h>
#include "DistributedLifeData.h"

DistributedLifeData::DistributedLifeData(const char *path) : LifeData(path) {
    MPI_Comm_size(MPI_COMM_WORLD, &this->size);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    std::cout << this->rank << "\n";
}

void DistributedLifeData::step() {

}
