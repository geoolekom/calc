//
// Created by geoolekom on 05.10.18.
//

#include "Master.h"
#include <mpi.h>

Master::Master() : size(0) {};

Master::Master(int size, Data* initialData): size(size) {
    data = new Data(*initialData);
    requests = new MPI_Request[size];
    statuses = new MPI_Status[size];
};

Master::~Master() {
    delete data;
    delete[] requests;
    delete[] statuses;
};

Data* Master::getData() {
    return this->data;
};

void Master::wait(int tag) {
    int nx = data->getNx(), ny = data->getNy(), count = ny * nx / size;
    for (int i = 0; i < size; i++) {
        MPI_Irecv(data->getData() + i * count, count, MPI_INT, i, tag, MPI_COMM_WORLD, requests + i);
    }
    MPI_Waitall(size, requests, statuses);
}
