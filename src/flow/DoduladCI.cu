//
// Created by geoolekom on 20.02.19.
//

#include "utils/cuda.h"
#include "DoduladCI.h"
#include <cuda_runtime_api.h>

DoduladCI::DoduladCI(double tStep, double vStep, int vGridRadius, State3D *state) : tStep(tStep), vStep(vStep), vGridRadius(vGridRadius) {
    ci::init(&potential, ci::Z_SYMM);
    vIndexMap = new IndexMap();
    for (const auto &vIndex: state->getVelocityIterable()) {
        auto svIndex = vIndex + intVector({state->nvx, state->nvy, state->nvz});
        (*vIndexMap)[svIndex[0] % state->nvx][svIndex[1] % state->nvy][svIndex[2] % state->nvz] = state->index({0, 0, 0}, vIndex);
    }
}

void DoduladCI::generateGrid(){
    ci::gen(tStep, KOROBOV_GRID_PARAMETER, vGridRadius, vGridRadius,
            *vIndexMap, *vIndexMap, vStep, particleMass, particleMass, particle, particle);
    size_t localNcSize = ci::nc.size();
    cudaMemcpy(&ncSize, &localNcSize, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &ncData, sizeof(ci::node_calc) * ncSize, cudaMemAttachGlobal);
    cudaMemcpy(ncData, ci::nc.data(), sizeof(ci::node_calc) * ncSize, cudaMemcpyHostToDevice);
}

void DoduladCI::finalizeGrid() {
    cudaFree(ncData);
}

__device__ void DoduladCI::calculateIntegral(double *slice) {
    ci::cudaIter(ncData, ncSize, slice, slice);
}

DoduladCI::~DoduladCI() {
    delete vIndexMap;
}
