//
// Created by geoolekom on 12.03.19.
//

#ifndef CALC_BASE_H
#define CALC_BASE_H

#include <array>

typedef std::array<int, 2> intVector;
typedef std::array<double, 2> doubleVector;

template <typename T>
double operator*(const std::array<double, 2>& one, const std::array<T, 2>& two) {
    return one[0] * two[0] + one[1] * two[1];
}

template <typename T>
std::array<T, 2> operator+(const std::array<T, 2>& one, const std::array<T, 2>& two) {
    return { one[0] + two[0], one[1] + two[1] };
}

template <typename T>
std::array<T, 2> operator-(const std::array<T, 2>& one, const std::array<T, 2>& two) {
    return { one[0] - two[0], one[1] - two[1] };
}

template <typename T>
std::array<double, 2> operator*(double k, const std::array<T, 2>& v) {
    return { k * v[0], k * v[1] };
}

template <typename T>
std::array<double, 2> operator/(const std::array<T, 2>& v, double k) {
    return { v[0] / k, v[1] / k };
}

#endif //CALC_BASE_H
