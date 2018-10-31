//
// Created by geoolekom on 30.10.18.
//

#include <mpi.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include "Data.h"

Data::Data(int rank, int size, int nx, double xRange, double highT, double lowT, double eps, int nvStart, int nvEnd)
: rank(rank), size(size), nx(nx), highT(highT), lowT(lowT), xStep(xRange / nx), eps(eps), nvStart(nvStart), nvEnd(nvEnd) {
    nv = nvEnd - nvStart;
    curr = new double[nx * nv];
    next = new double[nx * nv];

    temperature = new double[nx];
    prevTemperature = new double[nx];

    bufferNom = new double[nx * (size - 1)];
    bufferDenom = new double[nx * (size - 1)];
};

Data::~Data() {
    delete[] curr;
    delete[] next;
    delete[] temperature;
    delete[] prevTemperature;
    delete[] bufferNom;
    delete[] bufferDenom;
}

int Data::index(int xIndex, int vIndex) {
    return ((xIndex + nx) % nx) + ((vIndex - nvStart + nv) % nv) * nx;
}

double Data::getValue(int xIndex, int vIndex) {
    if (xIndex == 0) {
        return exp(- 0.5 * highT / lowT * vIndex * vIndex);
    } else if (xIndex + 1 == nx) {
        return exp(- 0.5 * vIndex * vIndex);
    } else {
        return curr[index(xIndex, vIndex)];
    }
}

double Data::deltaFunc(int xIndex, int vIndex) {
    double diff1 = fabs(getValue(xIndex, vIndex) - getValue(xIndex - 1, vIndex));
    double diff2 = fabs(getValue(xIndex + 1, vIndex) - getValue(xIndex, vIndex));
    double sgn = diff1 > 0 ? 1 : -1;
    return fmin(diff1, diff2) * sgn;
}

double Data::calcFunc(int xIndex, int vIndex) {
    double gamma = fabs(vIndex / 5.0);
    if (xIndex == 0) {
        return curr[index(xIndex, vIndex)];
    } else if (xIndex + 1 == nx) {
        return curr[index(xIndex, vIndex)];
    } else if (vIndex > 0) {
        return curr[index(xIndex, vIndex)] + (1 - gamma) * deltaFunc(xIndex, vIndex) / 2.0;
    } else {
        return curr[index(xIndex + 1, vIndex)] - (1 - gamma) * deltaFunc(xIndex + 1, vIndex) / 2.0;
    }
}

void Data::step(int stepNumber) {
    for (int vIndex = nvStart; vIndex < nvEnd; vIndex ++) {
        for (int xIndex = 0; xIndex < nx; xIndex ++) {
            next[index(xIndex, vIndex)] = calcFunc(xIndex, vIndex);
        }
    }
}

void Data::solve() {
    double* temp;
    int i = 0;
    while (true) {
        step(i);
        temp = curr;
        curr = next;
        next = temp;

        if (i % 100 == 0) {
            calcTemperature(i);
        } else if (i % 100 == 1) {
            temp = temperature;
            temperature = prevTemperature;
            prevTemperature = temp;
            calcTemperature(i);

            if (breakCondition()) {
                break;
            }
        }

        i ++;
    }
    std::cout << "Решено за " << i << " шагов.\n";
}

bool Data::breakCondition() {
    double maxDelta = 0;
    for (int xIndex = 0; xIndex < nx; xIndex ++) {
        double delta = fabs(temperature[xIndex] - prevTemperature[xIndex]) / temperature[xIndex];
        if (delta > maxDelta){
            maxDelta = delta;
        }
    }
    return nx * maxDelta < eps;
}

void Data::setInitialValues() {
    for (int vIndex = nvStart; vIndex < nvEnd; vIndex ++) {
        for (int xIndex = 0; xIndex < nx - 1; xIndex ++) {
            curr[index(xIndex, vIndex)] = exp(- 0.5 * highT / lowT * vIndex * vIndex);
        }
        curr[index(nx - 1, vIndex)] = exp(- 0.5 * vIndex * vIndex);
    }
}

void Data::calcTemperature(int stepNumber) {
    auto *denom = new double[nx]();
    auto *nom = new double[nx]();

    for (int vIndex = nvStart; vIndex < nvEnd; vIndex ++) {
        for (int xIndex = 0; xIndex < nx; xIndex ++) {
            nom[xIndex] += vIndex * vIndex * curr[index(xIndex, vIndex)];
            denom[xIndex] += curr[index(xIndex, vIndex)];
        }
    }

    if (rank == 0) {
        MPI_Request *requests;
        MPI_Status *statuses;
        if (size > 1) {
            requests = new MPI_Request[2 * (size - 1)];
            statuses = new MPI_Status[2 * (size - 1)];
            for (int i = 0; i < size - 1; i ++) {
                MPI_Irecv(bufferNom + i * nx, nx, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD, requests + i);
                MPI_Irecv(bufferDenom + i * nx, nx, MPI_DOUBLE, i + 1, 1, MPI_COMM_WORLD, requests + size - 1 + i);
            }

            MPI_Waitall(size, requests, statuses);
            for (int i = 0; i < size - 1; i ++) {
                for (int xIndex = 0; xIndex < nx; xIndex ++) {
                    denom[xIndex] += bufferDenom[i * nx + xIndex];
                    nom[xIndex] += bufferNom[i * nx + xIndex];
                }
            }
        }
        for (int xIndex = 0; xIndex < nx; xIndex ++) {
            temperature[xIndex] = highT * nom[xIndex] / denom[xIndex];
        }
    } else {
        MPI_Send(nom, nx, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(denom, nx, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Bcast(temperature, nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] denom;
    delete[] nom;
}
