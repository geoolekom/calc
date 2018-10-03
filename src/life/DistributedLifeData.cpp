//
// Created by geoolekom on 01.10.18.
//

#include <iostream>
#include <fstream>
#include <mpi.h>
#include "DistributedLifeData.h"

DistributedLifeData::DistributedLifeData(const char *path, int size, int rank) {
    this->size = size;
    this->rank = rank;

    std::ifstream file;
    file.open(path);

    file >> this->stepCount >> this->saveStep >> this->nx >> this->ny;
    u = new int[nx * ny / size];
    prev = new int[nx * ny / size];
    int i, j;
    while (file >> i >> j) {
        this->prev[this->index(i, j - rank * ny / size)] = 1;
    }
    dataLeft = new int[nx]();
    dataRight = new int[nx]();
    file.close();
}

int DistributedLifeData::index(int i, int j) {
    if (j == -1) {
        return 0;
    }
    return ((i + nx) % nx) + ((j + ny) % ny) * nx;
}

int DistributedLifeData::calculateStatus(int n, int oldValue) {
    if (n == 3 && oldValue == 0) {
        return  1;
    } else if ((n == 3 || n == 2) && oldValue == 1) {
        return  1;
    } else {
        return 0;
    }
}

void DistributedLifeData::step() {
    MPI_Request requestLeft, requestRight;
    int left = rank > 0 ? rank - 1 : size - 1;
    int right = rank + 1 < size ? rank + 1 : 0;

    MPI_Irecv(dataLeft, nx, MPI_INT, left, 0, MPI_COMM_WORLD, &requestLeft);
    MPI_Irecv(dataRight, nx, MPI_INT, right, 0, MPI_COMM_WORLD, &requestRight);

    MPI_Rsend(prev + rank * ny / size, nx, MPI_INT, left, 0, MPI_COMM_WORLD);
    MPI_Rsend(prev + (rank + 1) * ny / size, nx, MPI_INT, right, 0, MPI_COMM_WORLD);

    int i, j;
    for (j = 1; j < ny / size - 1; j++) {
        for (i = 0; i < nx; i++) {
            int n = cellStatus(i, j);
            this->u[this->index(i, j)] = calculateStatus(n, this->prev[this->index(i, j)]);
        }
    }
    MPI_Wait(&requestLeft, MPI_STATUS_IGNORE);
    for (i = 0; i < nx; i ++) {
        int n = cellStatus(i, 0);
        this->u[this->index(i, 0)] = calculateStatus(n, this->prev[this->index(i, 0)]);
    }
    MPI_Wait(&requestRight, MPI_STATUS_IGNORE);
    for (i = 0; i < nx; i ++) {
        int n = cellStatus(i, ny / size - 1);
        this->u[this->index(i, ny / size - 1)] = calculateStatus(n, this->prev[this->index(i, ny / size - 1)]);
    }
}

int DistributedLifeData::cellStatus(int i, int j) {
    int n = 0;
    if (j > 0) {
        n += this->prev[this->index(i - 1, j - 1)];
        n += this->prev[this->index(i, j - 1)];
        n += this->prev[this->index(i + 1, j - 1)];
    } else {
        n += dataLeft[i - 1];
        n += dataLeft[i];
        n += dataLeft[i + 1];
    }

    n += this->prev[this->index(i - 1, j)];
    n += this->prev[this->index(i + 1, j)];

    if (j + 1 < ny / size) {
        n += this->prev[this->index(i - 1, j + 1)];
        n += this->prev[this->index(i, j + 1)];
        n += this->prev[this->index(i + 1, j + 1)];
    } else {
        n += dataRight[i - 1];
        n += dataRight[i];
        n += dataRight[i + 1];
    }
    return n;
}

