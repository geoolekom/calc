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
        if (rank + 1 > j * size / nx > rank) {
            this->prev[this->index(i, j - rank * ny / size)] = 1;
        }
    }
    dataLeft = new int[nx]();
    dataRight = new int[nx]();
    file.close();
}

DistributedLifeData::~DistributedLifeData() {
    delete[] this->u;
    delete[] this->prev;
    delete[] this->dataLeft;
    delete[] this->dataRight;
}

int DistributedLifeData::index(int i, int j) {
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

void DistributedLifeData::step(int stepNumber) {
    MPI_Request requestLeft, requestRight;
    MPI_Status statusLeft, statusRight;
    int left = rank > 0 ? rank - 1 : size - 1;
    int right = rank + 1 < size ? rank + 1 : 0;

    MPI_Irecv(dataLeft, nx, MPI_INT, left, stepNumber, MPI_COMM_WORLD, &requestLeft);
    MPI_Irecv(dataRight, nx, MPI_INT, right, stepNumber, MPI_COMM_WORLD, &requestRight);

    MPI_Rsend(prev + rank * ny / size, nx, MPI_INT, left, stepNumber, MPI_COMM_WORLD);
    MPI_Rsend(prev + (rank + 1) * ny / size, nx, MPI_INT, right, stepNumber, MPI_COMM_WORLD);

    int i, j;
    for (j = 1; j < ny / size - 1; j++) {
        for (i = 0; i < nx; i++) {
            int n = cellStatus(i, j);
            this->u[this->index(i, j)] = calculateStatus(n, this->prev[this->index(i, j)]);
        }
    }
    MPI_Wait(&requestLeft, &statusLeft);
    for (i = 0; i < nx; i ++) {
        int n = cellStatus(i, 0);
        this->u[this->index(i, 0)] = calculateStatus(n, this->prev[this->index(i, 0)]);
    }
    MPI_Wait(&requestRight, &statusRight);
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

void DistributedLifeData::toFile(const char* path) {
    std::ofstream file;
    file.open(path);

    file << "# vtk DataFile Version 3.0\n";
    file << "Created by write_to_vtk2d\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << this->nx + 1 << " " << this->ny + 1 << " 1\n";
    file << "SPACING 1 1 0.0\n";
    file << "ORIGIN 0 0 0.0\n";
    file << "CELL_DATA " << this->nx * this->ny << "\n";

    file << "SCALARS life int 1\n";
    file << "LOOKUP_TABLE life_table\n";
    for (int i = 0; i < this->ny; i++) {
        for (int j = 0; j < this->nx; j++) {
            file << this->u[this->index(j, i)] << "\n";
        }
    }
    file.close();
}

void DistributedLifeData::evolve() {
    char path[100];
    for (int i = 0; i < this->stepCount; i++) {
        if (i % this->saveStep == 0 && rank == 0) {
            sprintf(path, "data/life/life_%06d.vtk", i);
            this->toFile(path);
        }
        this->step(i);
        int *tmp;
        tmp = this->prev;
        this->prev = this->u;
        this->u = tmp;
    }
}

