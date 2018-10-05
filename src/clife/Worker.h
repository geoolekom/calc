//
// Created by geoolekom on 05.10.18.
//

#ifndef CALC_WORKER_H
#define CALC_WORKER_H

#include "Data.h"


class Worker {
private:
    int rank, size;
    Data *curr, *prev;
    int *bufferLeft, *bufferRight;
public:
    Worker();
    Worker(int rank, int size, Data* initialData);
    ~Worker();
    void makeStep(int tag);
    void sendData(int tag);

    int calculateValue(int i, int j);
};


#endif //CALC_WORKER_H
