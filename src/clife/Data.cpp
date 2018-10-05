//
// Created by geoolekom on 05.10.18.
//

#include <fstream>
#include "Data.h"

Data::Data() {
    this->array = new int[0];
}

Data::Data(int nx, int ny, int *initialArray) : nx(nx), ny(ny) {
    this->array = new int[nx * ny];
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            this->array[this->index(j, i)] = initialArray[this->index(j, i)];
        }
    }
};

Data::Data(const Data& anotherData) : Data(anotherData.getNx(), anotherData.getNy(), anotherData.getData()) {};

Data::~Data() {
    delete[] this->array;
};

int Data::index(int xCounter, int yCounter) {
    return ((xCounter + nx) % nx) + ((yCounter + ny) % ny) * nx;
}

void Data::setValue(int xCounter, int yCounter, int value) {
    array[this->index(xCounter, yCounter)] = value;
}

int Data::getValue(int xCounter, int yCounter) {
    return array[this->index(xCounter, yCounter)];
}

void Data::toFile(const char *path) {
    std::ofstream file;
    file.open(path);

    file << "# vtk DataFile Version 3.0\n";
    file << "Created by write_to_vtk2d\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " 1\n";
    file << "SPACING 1 1 0.0\n";
    file << "ORIGIN 0 0 0.0\n";
    file << "CELL_DATA " << nx * ny << "\n";

    file << "SCALARS life int 1\n";
    file << "LOOKUP_TABLE life_table\n";
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            file << this->getValue(j, i) << "\n";
        }
    }
    file.close();
}
