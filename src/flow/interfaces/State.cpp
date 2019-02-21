//
// Created by geoolekom on 20.02.19.
//

#include "State.h"

PointIterator<SpatialPoint, Velocity> SpatialPoint::begin() {
    return PointIterator<SpatialPoint, Velocity>(this, Velocity(this->data));
}

PointIterator<SpatialPoint, Velocity> SpatialPoint::end() {
    return PointIterator<SpatialPoint, Velocity>(this, Velocity(this->data + this->size));
}
