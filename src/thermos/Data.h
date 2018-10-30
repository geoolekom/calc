//
// Created by geoolekom on 30.10.18.
//

#ifndef CALC_DATA_H
#define CALC_DATA_H


class Data {
private:
    double xStep, vStep, tStep;
    int nx, nv;
    double *curr, *next;
    double eps;
    double lowT, highT;
    double *temperature, *prevTemperature;
public:
    Data();
    Data(int nx, double xRange, double highT, double lowT, double eps);
    ~Data();
    int index(int xIndex, int vIndex);
    double getValue(int xIndex, int vIndex);

    void setInitialValues();

    double deltaFunc(int xIndex, int vIndex);
    double calcFunc(int xIndex, int vIndex);
    void step(int stepNumber);
    void solve();
    bool breakCondition();

    void toFile(const char* filename);
    void calcTemperature();
};


#endif //CALC_DATA_H
