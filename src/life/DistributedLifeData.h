//
// Created by geoolekom on 01.10.18.
//

#ifndef CALC_DISTRIBUTEDLIFEDATA_H
#define CALC_DISTRIBUTEDLIFEDATA_H


#include "LifeData.h"

class DistributedLifeData : public LifeData {
private:
    int rank, size;
public:
    DistributedLifeData(const char* path);
    void step();
};


#endif //CALC_DISTRIBUTEDLIFEDATA_H
