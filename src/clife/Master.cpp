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
    int nx = data->getNx(), ny = data->getNy();
    int count = ny * nx / size;
    int countFirst = (ny / size + ny % size) * nx;
    MPI_Irecv(data->getData(), countFirst, MPI_INT, 0, tag, MPI_COMM_WORLD, requests);
    for (int i = 1; i < size; i++) {
        MPI_Irecv(data->getData() + i * count, count, MPI_INT, i, tag, MPI_COMM_WORLD, requests + i);
    }
    MPI_Waitall(size, requests, statuses);
}
