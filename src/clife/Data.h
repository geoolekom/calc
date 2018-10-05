//
// Created by geoolekom on 05.10.18.
//

#ifndef CALC_DATA_H
#define CALC_DATA_H


class Data {
private:
    int nx, ny;
    int *array;
    int index(int xCounter, int yCounter);
public:
    Data();
    Data(int nx, int ny, int* data);
    Data(const Data&);
    ~Data();

    void setValue(int xCounter, int yCounter, int value);
    int getValue(int xCounter, int yCounter);

    int getNx() const { return nx; };
    int getNy() const { return ny; };
    int* getData() const { return array; };

    void toFile(const char* path);
};


#endif //CALC_DATA_H
