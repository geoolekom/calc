//
// Created by geoolekom on 30.10.18.
//

#include "Data.h"
#include <cmath>
#include <iostream>
#include <fstream>

Data::Data(int nx, double xRange, double highT, double lowT, double eps) : nx(nx), highT(highT), lowT(lowT), xStep(xRange / nx), eps(eps) {
    nv = 5;
    curr = new double[nx * 2 * nv];
    next = new double[nx * 2 * nv];

    temperature = new double[nx];
    prevTemperature = new double[nx];
};

Data::~Data() {
    delete[] curr;
    delete[] next;
    delete[] temperature;
    delete[] prevTemperature;
}

int Data::index(int xIndex, int vIndex) {
    return ((xIndex + nx) % nx) + ((vIndex + 2 * nv) % (2 * nv)) * nx;
}

double Data::getValue(int xIndex, int vIndex) {
    if (xIndex == 0) {
        return exp(- 0.5 * vIndex * vIndex);
    } else if (xIndex + 1 == nx) {
        return exp(- 0.5 * highT / lowT * vIndex * vIndex);
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
    for (int vIndex = - nv + 1; vIndex < nv; vIndex ++) {
        for (int xIndex = 0; xIndex < nx; xIndex ++) {
            next[index(xIndex, vIndex)] = calcFunc(xIndex, vIndex);
        }
    }
}

void Data::solve() {
    double* temp;
    int i = 0;
    while (!breakCondition()) {
        step(i);
        temp = curr;
        curr = next;
        next = temp;

        calcTemperature();
        temp = temperature;
        temperature = prevTemperature;
        prevTemperature = temp;
        i ++;
    }
    std::cout << "Solved in " << i << " steps.\n";
}

bool Data::breakCondition() {
    calcTemperature();
    double maxDelta = 0;
    for (int xIndex = 1; xIndex < nx; xIndex ++) {
        double delta = (temperature[xIndex] - prevTemperature[xIndex]) / temperature[xIndex];
        if (delta > maxDelta){
            maxDelta = delta;
        }
    }
    return maxDelta < eps;
}

void Data::setInitialValues() {
    for (int vIndex = - nv + 1; vIndex < nv; vIndex ++) {
        for (int xIndex = 0; xIndex < nx - 1; xIndex ++) {
            curr[index(xIndex, vIndex)] = exp(- 0.5 * highT / lowT * vIndex * vIndex);
        }
        curr[index(nx - 1, vIndex)] = exp(- 0.5 * vIndex * vIndex);
    }
}

void Data::toFile(const char *filename) {
    std::ofstream file;
    file.open(filename);
    for (int xIndex = 0; xIndex < nx; xIndex ++) {
        file << (double) xIndex * xStep << "\t" << temperature[xIndex] << std::endl;
    }
    file.close();
}

void Data::calcTemperature() {
    auto *denom = new double[nx];
    auto *nom = new double[nx];

    for (int vIndex = - nv + 1; vIndex < nv; vIndex ++) {
        for (int xIndex = 0; xIndex < nx; xIndex ++) {
            denom[xIndex] += curr[index(xIndex, vIndex)];
            nom[xIndex] += vIndex * vIndex * curr[index(xIndex, vIndex)];
        }
    }
    for (int xIndex = 0; xIndex < nx; xIndex ++) {
        temperature[xIndex] = highT * nom[xIndex] / denom[xIndex];
    }
    delete[] denom;
    delete[] nom;
}