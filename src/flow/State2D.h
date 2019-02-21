#ifndef CALC_STATE2D_H
#define CALC_STATE2D_H


class State2D {
private:
    double* data;
public:
    int nvx, nvy, nx, ny;
    int xIndexMax, yIndexMax, vxIndexMin, vxIndexMax, vyIndexMin, vyIndexMax;

    inline double getValue(int xIndex, int yIndex, int vxIndex, int vyIndex) {
        return data[this->index(xIndex, yIndex, vxIndex, vyIndex)];
    };

    inline void setValue(int xIndex, int yIndex, int vxIndex, int vyIndex, double value) {
        data[this->index(xIndex, yIndex, vxIndex, vyIndex)] = value;
    };

    State2D(int xIndexMax, int yIndexMax, int vxIndexMin, int vxIndexMax, int vyIndexMin, int vyIndexMax) :
        xIndexMax(xIndexMax), yIndexMax(yIndexMax),
        vxIndexMin(vxIndexMin), vxIndexMax(vxIndexMax),
        vyIndexMin(vyIndexMin), vyIndexMax(vyIndexMax) {
        this->nx = xIndexMax;
        this->ny = yIndexMax;
        this->nvx = vxIndexMax - vxIndexMin;
        this->nvy = vyIndexMax - vyIndexMin;
        this->data = new double[nx * ny * nvx * nvy]();
    }

    State2D(const State2D& state) :
            xIndexMax(state.xIndexMax), yIndexMax(state.yIndexMax),
            vxIndexMin(state.vxIndexMin), vxIndexMax(state.vxIndexMax),
            vyIndexMin(state.vyIndexMin), vyIndexMax(state.vyIndexMax) {
        this->nx = state.xIndexMax;
        this->ny = state.yIndexMax;
        this->nvx = state.vxIndexMax - state.vxIndexMin;
        this->nvy = state.vyIndexMax - state.vyIndexMin;
        this->data = new double[nx * ny * nvx * nvy]();
    }

    ~State2D() {
        delete this->data;
    };

    inline int index(int xIndex, int yIndex, int vxIndex, int vyIndex) {
        return (vxIndex + nvx) % nvx
            + ((vyIndex + nvy) % nvy) * nvx
            + ((xIndex + nx) % nx) * nvx * nvy
            + ((yIndex + ny) % ny) * nvx * nvy * nx;
    }

    inline double* velocitySlice(int xIndex, int yIndex) {
        return data + nvx * nvy * (yIndex + xIndex * ny);
    }
};


#endif //CALC_STATE2D_H
