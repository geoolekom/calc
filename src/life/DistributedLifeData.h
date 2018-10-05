//
// Created by geoolekom on 01.10.18.
//

#ifndef CALC_DISTRIBUTEDLIFEDATA_H
#define CALC_DISTRIBUTEDLIFEDATA_H


class DistributedLifeData {
private:
    int size, rank;
    int* dataLeft;
    int* dataRight;

    int nx, ny;
    int *u, *prev;
    int stepCount, saveStep;
protected:
    int index(int i, int j);
    void step(int stepNumber);
    int cellStatus(int i, int j);
    int calculateStatus(int n, int oldStatus);
public:
    DistributedLifeData(const char* path, int size, int rank);
    ~DistributedLifeData();
    void toFile(const char* path);
    void evolve();
};


#endif //CALC_DISTRIBUTEDLIFEDATA_H
