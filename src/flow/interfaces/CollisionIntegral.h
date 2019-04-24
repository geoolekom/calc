//
// Created by geoolekom on 21.02.19.
//

#ifndef CALC_COLLISIONINTEGRAL_H
#define CALC_COLLISIONINTEGRAL_H

template <typename State>
class CollisionIntegral {
public:
    CollisionIntegral() = default;
    virtual void stepForward() {};
    virtual void calculateIntegral(State* state, int xIndex, int yIndex) {};
};

#endif //CALC_COLLISIONINTEGRAL_H
