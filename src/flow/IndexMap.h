//
// Created by geoolekom on 18.02.19.
//

#ifndef CALC_INDEXMAP_H
#define CALC_INDEXMAP_H


class IndexMap {
private:
    int** data;
    int nvx, nvy;
public:
    IndexMap(int nvx, int nvy) : nvx(nvx), nvy(nvy) {
        this->data = new int*[nvx];
        for (int i = 0; i < nvx; i++) {
            this->data[i] = new int[nvy];
            for (int j = 0; j < nvy; j++) {
                this->data[i][j] = i + nvx * j;
            }
        }
    }

    ~IndexMap() {
        for (int i = 0; i < nvx; i++) {
            delete[] this->data[i];
        }
        delete[] this->data;
    }

    int** operator[] (int i) const {
        return data;
    }
};


#endif //CALC_INDEXMAP_H
