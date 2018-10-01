//
// Created by geoolekom on 01.10.18.
//

#include <iostream>
#include <fstream>
#include "LifeData.h"


LifeData::LifeData(const char *path) {
    std::ifstream file;
    file.open(path);

    file >> this->stepCount >> this->saveStep >> this->nx >> this->ny;
    std::cout << "Number of steps: " << this->stepCount << ". Save every " << this->saveStep << " step.\n";
    this->u = new int[nx * ny];
    this->prev = new int[nx * ny];
    int i, j;
    while (file >> i >> j) {
        this->prev[this->index(i, j)] = 1;
    }
    file.close();
}

LifeData::~LifeData() {
    delete[] this->u;
    delete[] this->prev;
}


int LifeData::index(int i, int j) {
    return ((i + this->nx) % this->nx) + ((j + this->ny) % this->ny) * (this->nx);
}

void LifeData::step() {
    int i, j;
    for (j = 0; j < this->ny; j++) {
        for (i = 0; i < this->nx; i++) {
            int n = 0;
            n += this->prev[this->index(i+1, j)];
            n += this->prev[this->index(i+1, j+1)];
            n += this->prev[this->index(i,   j+1)];
            n += this->prev[this->index(i-1, j)];
            n += this->prev[this->index(i-1, j-1)];
            n += this->prev[this->index(i,   j-1)];
            n += this->prev[this->index(i-1, j+1)];
            n += this->prev[this->index(i+1, j-1)];
            this->u[this->index(i, j)] = 0;
            if (n == 3 && this->prev[this->index(i,j)] == 0) {
                this->u[this->index(i,j)] = 1;
            }
            if ((n == 3 || n == 2) && this->prev[this->index(i,j)] == 1) {
                this->u[this->index(i,j)] = 1;
            }
        }
    }
}

void LifeData::toFile(const char *path) {
    std::ofstream file;
    file.open(path);

    file << "# vtk DataFile Version 3.0\n";
    file << "Created by write_to_vtk2d\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << this->nx+1 << " " << this->ny+1 << " 1\n";
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

void LifeData::evolve() {
    char path[100];
    for (int i = 0; i < this->stepCount; i++) {
        if (i % this->saveStep == 0) {
            sprintf(path, "data/life/life_%06d.vtk", i);
            std::cout << "Saving step " << i << " to '" << path << "'.\n";
            this->toFile(path);
        }
        this->step();
        int *tmp;
        tmp = this->prev;
        this->prev = this->u;
        this->u = tmp;
    }
}
