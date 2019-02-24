//
// Created by geoolekom on 21.02.19.
//

#ifndef CALC_COLLISIONINTEGRAL_H
#define CALC_COLLISIONINTEGRAL_H

template <typename State>
class CollisionIntegral {
public:
    virtual void stepForward() = 0;
    virtual void calculateIntegral(State* state, int xIndex, int yIndex) = 0;
};

#endif //CALC_COLLISIONINTEGRAL_H
