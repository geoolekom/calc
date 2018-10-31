//
// Created by geoolekom on 30.10.18.
//

#ifndef CALC_DATA_H
#define CALC_DATA_H


class Data {
private:
    int rank, size;
    double xStep;
    int nx, nv, nvStart, nvEnd;
    double *curr, *next;
    double eps;
    double lowT, highT;
    double *temperature, *prevTemperature;
    double *bufferNom, *bufferDenom;
public:
    Data();
    Data(int rank, int size, int nx, double xRange, double highT, double lowT, double eps, int nvStart, int nvEnd);
    ~Data();
    int index(int xIndex, int vIndex);
    double getValue(int xIndex, int vIndex);

    int getNx() { return nx; }
    double getXStep() { return xStep; }
    double* getData() { return curr; }
    double* getTemperature() { return temperature; }

    void setInitialValues();

    double deltaFunc(int xIndex, int vIndex);
    double calcFunc(int xIndex, int vIndex);
    void step(int stepNumber);
    void solve();
    bool breakCondition();

    void calcTemperature(int stepNumber);
};


#endif //CALC_DATA_H
