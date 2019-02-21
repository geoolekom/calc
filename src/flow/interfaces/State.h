//
// Created by geoolekom on 20.02.19.
//

#ifndef CALC_STATE_H
#define CALC_STATE_H

#include <iostream>


template <typename StateType, typename PointType>
class PointIterator {
private:
    PointType value;
    StateType* state;
public:
    typedef PointIterator<StateType, PointType> selfType;

    PointIterator(StateType* state, const PointType& p) : state(state), value(p) {};
    PointType& operator*() { return value; };
    selfType& operator++() {
        this->value = state->populatePoint(this->value.getData() + this->value.getSize());
        return *this;
    };
    bool operator==(const selfType& other) { return this->value.getData() == other.value.getData(); };
    bool operator!=(const selfType& other) { return this->value.getData() != other.value.getData(); };
    bool operator<(const selfType& other) { return this->value.getData() - other.value.getData() < 0; };
};


class Vector {
public:
    virtual double operator* (const Vector& v) const { return 0; };
};


class Point {
protected:
    double* data;
    long int size;
public:
    Point(double *data, long int size = 1) : data(data), size(size) {};
    ~Point() {};
    double* getData() const { return this->data; };
    long int getSize() const { return this->size; };

};


class Velocity : public Point, public Vector {
public:
    Velocity(double* data, long int size = 1) : Point(data, size) {};
};


template <typename VelocityType>
class SpatialPoint : public Point, public Vector {
protected:
    virtual VelocityType populatePoint(double* data) {
        return VelocityType(data);
    }
    friend class PointIterator<SpatialPoint, VelocityType>;
public:
    typedef VelocityType velocityType;

    SpatialPoint(double* data, long int size = 1) : Point(data, size) {};

    PointIterator<SpatialPoint, VelocityType> begin() {
        return PointIterator<SpatialPoint, VelocityType>(this, this->populatePoint(this->data));
    };

    PointIterator<SpatialPoint, VelocityType> end() {
        return PointIterator<SpatialPoint, VelocityType>(this, this->populatePoint(this->data + this->size));
    };
};


template <typename SpatialPointType>
class State : public Point {
protected:
    virtual SpatialPointType populatePoint(double* data) {
        return SpatialPointType(data);
    }
    friend class PointIterator<State, SpatialPointType>;
public:
    typedef SpatialPointType spatialPointType;
    typedef typename SpatialPointType::velocityType velocityType;

    State(double* data, long int size = 1) : Point(data, size) {};

    PointIterator<State, SpatialPointType> begin() {
        return PointIterator<State, SpatialPointType>(this, this->populatePoint(this->data));
    };

    PointIterator<State, SpatialPointType> end() {
        return PointIterator<State, SpatialPointType>(this, this->populatePoint(this->data + this->size));
    };

    double* velocitySlice(const SpatialPointType& point) {
        return point.getData();
    };

    double getValue(const SpatialPointType& point, const velocityType& velocity) {
        return *(velocity.getData());
    };

    void setValue(SpatialPointType& point, velocityType& velocity, double value) {
        *(velocity.getData()) = value;
    };
};


#endif //CALC_STATE_H
