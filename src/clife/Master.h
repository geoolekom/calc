//
// Created by geoolekom on 05.10.18.
//

#ifndef CALC_MASTER_H
#define CALC_MASTER_H


#include <mpi.h>
#include "Data.h"

class Master {
private:
    Data* data;
    int size;
    MPI_Request* requests;
    MPI_Status* statuses;
public:
    Master();
    Master(int size, Data* initialData);
    ~Master();
    void wait(int tag);
    Data* getData();
};


#endif //CALC_MASTER_H
