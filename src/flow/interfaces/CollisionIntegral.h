//
// Created by geoolekom on 21.02.19.
//

#include <cuda_runtime_api.h>

#ifndef CALC_COLLISIONINTEGRAL_H
#define CALC_COLLISIONINTEGRAL_H

template <typename State>
class CollisionIntegral {
public:
    CollisionIntegral() = default;
    virtual void stepForward() {};
    __device__ virtual void calculateIntegral(double* slice) {};
};

#endif //CALC_COLLISIONINTEGRAL_H
