//
// Created by geoolekom on 21.02.19.
//

#include "interfaces/State.h"
#include "interfaces/Evolution.h"


class Velocity2D : public Velocity {
public:
    Velocity2D(double *data, long int size = 1) : Velocity(data, size) {};
    std::string toString() {
        return "bbbbb";
    }
};


class Point2D : public SpatialPoint<Velocity2D> {
public:
    Point2D(double *data, long int size = 1) : SpatialPoint<Velocity2D>(data, size) {};
    std::string toString() {
        return "aaaaa";
    }
};


class State2D : public State<Point2D> {
private:
    int nx = 2, ny = 2, nvx = 1, nvy = 1;
public:
    State2D(double *data, long int size = 1) : State(data, size) {};
};


class Evolution2D : public Evolution<State2D> {
public:
    Evolution2D(State2D** state) : Evolution<State2D>(state) {};

    double calculateFlowFactor(const spatialPointType& p) override {
        return 1.0;
    };

    double calculateDistributionFunction(const spatialPointType& p, const velocityType& v) override {
        return 1.0;
    }
};

int main(int argc, char* argv[]) {
    long int size = 2 * 2 * 1 * 1;
    double* data = new double[size]();
    auto state = new State2D(data, size);
    auto evolution = new Evolution2D(&state);
    evolution->evolve(15);

    delete evolution;
    delete state;
    delete data;
    return 0;
}