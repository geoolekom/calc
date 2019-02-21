//
// Created by geoolekom on 20.02.19.
//

#ifndef CALC_STATE_H
#define CALC_STATE_H


template <typename StateType, typename PointType>
class PointIterator {
private:
    PointType value;
    StateType* state;
public:
    typedef PointIterator<StateType, PointType> selfType;

    PointIterator(StateType* state, const PointType& p) : state(state), value(p) {};
    PointType& operator*() { return value; };
    selfType operator++() { return selfType(state, this->value.getData() + this->value.getSize()); };
    bool operator==(const selfType& other) { return this->value.getData() == other.value.getData(); };
    bool operator!=(const selfType& other) { return this->value.getData() != other.value.getData(); };
};


class Vector {
public:
    virtual double operator* (const Vector& p) const { return 0; };
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


class SpatialPoint : public Point, public Vector {
public:
    SpatialPoint(double* data, long int size = 1) : Point(data, size) {};
    PointIterator<SpatialPoint, Velocity> begin();
    PointIterator<SpatialPoint, Velocity> end();
};


template <typename SpatialPointType, typename VelocityType>
class State : public Point {
public:
    State(double* data, long int size = 1) : Point(data, size) {};

    PointIterator<State, SpatialPointType> begin() {
        return PointIterator<State, SpatialPointType>(this, SpatialPoint(this->data));
    };
    PointIterator<State, SpatialPointType> end() {
        return PointIterator<State, SpatialPointType>(this, SpatialPoint(this->data + this->size));
    };

    virtual double* velocitySlice(const SpatialPointType& point) = 0;
    virtual double getValue(const SpatialPointType& point, const VelocityType& velocity) = 0;
    virtual void setValue(SpatialPointType& point, VelocityType& velocity, double value) = 0;
};


#endif //CALC_STATE_H
