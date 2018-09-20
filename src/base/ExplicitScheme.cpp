//
// Created by geoolekom on 21.09.18.
//

#include <fstream>
#include <cmath>
#include "ExplicitScheme.h"

void ExplicitScheme::solve(int t) {
    this->setInitialValues();
    for (int i = 0; i < t; i++) {
        step();
    }
}

ExplicitScheme::ExplicitScheme(int size, int c) {
    this->size = size;
    this->c = c;
    this->prev = new double[size];
    this->u = new double[size];
}

ExplicitScheme::~ExplicitScheme() {
    delete[] this->prev;
    delete[] this->u;
}

void ExplicitScheme::toFile(char *filename) {
    std::ofstream file;
    file.open(filename);
    for (int i = 0; i < this->size; i++) {
        file << (double) i / this->size << "\t" << this->prev[i] << std::endl;
    }
    file.close();
}

void ExplicitScheme::setInitialValues(){
    for (int i = 0; i < this->size; i ++) {
        this->prev[i] = exp(-((double) i / size - 0.2)*((double) i / size - 0.2)/0.0025);
    }
}