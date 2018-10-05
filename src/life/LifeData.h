//
// Created by geoolekom on 01.10.18.
//

#ifndef CALC_LIFEDATA_H
#define CALC_LIFEDATA_H


class LifeData {
private:
    int nx, ny;
    int *u, *prev;
    int stepCount, saveStep;

    virtual int index(int i, int j);
    virtual void step();
    virtual int cellStatus(int i, int j);
public:
    explicit LifeData() {};
    explicit LifeData(const char* path);
    virtual ~LifeData();
    virtual void toFile(const char* path);
    void evolve();

};


#endif //CALC_LIFEDATA_H
