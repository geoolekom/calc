//
// Created by geoolekom on 05.10.18.
//

#include "Worker.h"
#include <mpi.h>

Worker::Worker() {
    prev = new Data();
    curr = new Data();
}

Worker::Worker(int rank, int size, Data *initialData) : rank(rank), size(size) {
    int nx = initialData->getNx(), ny = initialData->getNy();
    int newNy = ny / size;
    if (ny % size != 0 && rank == 0) {
        newNy += ny % size;
    }
    prev = new Data(nx, newNy, initialData->getData() + nx * ny * rank / size);
    curr = new Data(*prev);
    bufferLeft = new int[nx]();
    bufferRight = new int[nx]();
}

Worker::~Worker() {
    delete prev;
    delete curr;
    delete[] bufferLeft;
    delete[] bufferRight;
}

void Worker::makeStep(int stepNumber) {
    MPI_Request requestLeft, requestRight;
    MPI_Status statusLeft, statusRight;
    int left = rank > 0 ? rank - 1 : size - 1;
    int right = rank + 1 < size ? rank + 1 : 0;
    int nx = prev->getNx(), ny = prev->getNy();

    MPI_Irecv(bufferLeft, nx, MPI_INT, left, stepNumber, MPI_COMM_WORLD, &requestLeft);
    MPI_Irecv(bufferRight, nx, MPI_INT, right, stepNumber, MPI_COMM_WORLD, &requestRight);

    MPI_Rsend(prev->getData(), nx, MPI_INT, left, stepNumber, MPI_COMM_WORLD);
    MPI_Rsend(prev->getData() + nx * (ny - 1), nx, MPI_INT, right, stepNumber, MPI_COMM_WORLD);

    for (int j = 1; j < ny - 1; j++) {
        for (int i = 0; i < nx; i++) {
            int value = calculateValue(i, j);
            curr->setValue(i, j, value);
        }
    }
    MPI_Wait(&requestLeft, &statusLeft);
    for (int i = 0; i < nx; i++) {
        int value = calculateValue(i, 0);
        curr->setValue(i, 0, value);
    }
    MPI_Wait(&requestRight, &statusRight);
    for (int i = 0; i < nx; i++) {
        int value = calculateValue(i, ny - 1);
        curr->setValue(i, ny - 1, value);
    }
    Data* temp = prev;
    prev = curr;
    curr = temp;
}

int Worker::calculateValue(int i, int j) {
    int n = 0;
    if (j > 0) {
        n += prev->getValue(i - 1, j - 1);
        n += prev->getValue(i, j - 1);
        n += prev->getValue(i + 1, j - 1);
    } else {
        n += bufferLeft[i - 1];
        n += bufferLeft[i];
        n += bufferLeft[i + 1];
    }

    n += prev->getValue(i - 1, j);
    n += prev->getValue(i + 1, j);

    if (j + 1 < prev->getNy()) {
        n += prev->getValue(i - 1, j + 1);
        n += prev->getValue(i, j + 1);
        n += prev->getValue(i + 1, j + 1);
    } else {
        n += bufferRight[i - 1];
        n += bufferRight[i];
        n += bufferRight[i + 1];
    }

    int oldValue = prev->getValue(i, j);
    if (n == 3 && oldValue == 0) {
        return  1;
    } else if ((n == 3 || n == 2) && oldValue == 1) {
        return  1;
    } else {
        return 0;
    }
}


void Worker::sendData(int tag) {
    int nx = prev->getNx(), ny = prev->getNy();
    MPI_Ssend(prev->getData(), nx * ny, MPI_INT, size, tag, MPI_COMM_WORLD);
}
