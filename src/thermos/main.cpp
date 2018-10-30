//
// Created by geoolekom on 25.10.18.
//

#include <iostream>
#include "Data.h"

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cout << "Недостаточно аргументов.\n";
        return 1;
    }
    char *endptr;
    int nx = (int) strtol(argv[1], &endptr, 10);
    double xRange = strtod(argv[2], &endptr),
           highT = strtod(argv[3], &endptr),
           lowT = strtod(argv[4], &endptr),
           eps = strtod(argv[5], &endptr);
    char *outputFile = argv[6];
    auto* data = new Data(nx, xRange, highT, lowT, eps);
    data->setInitialValues();
    data->solve();
    data->calcTemperature();
    data->toFile(outputFile);
    return 0;
}

