//
// Created by geoolekom on 27.04.19.
//

#ifndef CALC_EVOLUTION3D_H
#define CALC_EVOLUTION3D_H

#include <cuda_runtime_api.h>
#include "State3D.h"
#include "Grid3D.h"
#include "DoduladCI.h"
#include "IndexTankWithScreen2D.cu"

class Evolution3D {
private:
    double tStep;

    Geometry3D* geometry;
    Grid3D* grid;
    DoduladCI* ci;

    State3D* curr;
    State3D* prev;
    State3D* state;

    floatType* currData;
    floatType* prevData;
public:
    Evolution3D(double tStep, State3D* state, Grid3D* grid, Geometry3D* geometry, DoduladCI* ci);
    ~Evolution3D();

    State3D* getState() const;
    void exportToHost();
    void swap();

    __device__ double calculateDiffusionFactor(const intVector& xIndex, const intVector& direction);
    __device__ void makeStep(int step, int txindex, int tyIndex, int tzIndex, int txStep, int tyStep, int tzStep);
    __device__ inline double limiter(double theta);
    __device__ inline double limitValue(double gamma, const intVector& direction, const intVector& xIndex, const intVector& vIndex);
    __device__ double schemeChange(const intVector& xIndex, int vxIndex, int vyIndex, int vzIndex);

};


#endif //CALC_EVOLUTION3D_H
