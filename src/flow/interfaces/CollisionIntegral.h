//
// Created by geoolekom on 21.02.19.
//

#include <host_defines.h>

#ifndef CALC_COLLISIONINTEGRAL_H
#define CALC_COLLISIONINTEGRAL_H

template <typename State>
class CollisionIntegral {
public:
    CollisionIntegral() = default;
    __device__ virtual void stepForward() {};
    __device__ virtual void calculateIntegral(State* state, int xIndex, int yIndex) {};
};

#endif //CALC_COLLISIONINTEGRAL_H
