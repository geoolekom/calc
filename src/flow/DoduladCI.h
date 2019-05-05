//
// Created by geoolekom on 20.02.19.
//

#ifndef CALC_DODULADCI_H
#define CALC_DODULADCI_H

#include <map>
#include <dodulad_ci/ci.hpp>
#include "State3D.h"
#include "interfaces/CollisionIntegral.h"

#define KOROBOV_GRID_PARAMETER 50000

class DoduladCI {
private:
    typedef std::map<int, std::map<int, std::map<int, int> > > IndexMap;
    double particleDiameter = 1.0, particleMass = 1.0, vStep, tStep;
    int vGridRadius;
    ci::HSPotential potential;
    ci::Particle particle = {particleDiameter};

    IndexMap* vIndexMap;

    ci::node_calc* ncData;
    size_t ncSize;
public:
    DoduladCI(double tStep, double vStep, int vGridRadius, State3D* state);

    ~DoduladCI();

    void generateGrid();
    void finalizeGrid();

    __device__ void calculateIntegral(double* slice);
};


#endif //CALC_DODULADCI_H
