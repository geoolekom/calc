//
// Created by geoolekom on 01.10.18.
//

#ifndef CALC_LIFEDATA_H
#define CALC_LIFEDATA_H


class LifeData {
protected:
    int nx, ny;
    int *u, *prev;
    int stepCount, saveStep;
    int index(int i, int j);
    virtual void step();
public:
    explicit LifeData(const char* path);
    ~LifeData();
    void toFile(const char* path);
    void evolve();

};


#endif //CALC_LIFEDATA_H
