//
// Created by geoolekom on 01.10.18.
//

#ifndef CALC_DISTRIBUTEDLIFEDATA_H
#define CALC_DISTRIBUTEDLIFEDATA_H


#include "LifeData.h"

class DistributedLifeData : public LifeData {
private:
    int size, rank;
    int* dataLeft;
    int* dataRight;
protected:
    int index(int i, int j) override;
    void step() override;
    int cellStatus(int i, int j) override;
    int calculateStatus(int n, int oldStatus);
public:
    DistributedLifeData(const char* path, int size, int rank);
};


#endif //CALC_DISTRIBUTEDLIFEDATA_H
