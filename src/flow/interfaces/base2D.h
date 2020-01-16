//
// Created by geoolekom on 12.03.19.
//

#ifndef CALC_BASE2D_H
#define CALC_BASE2D_H

#include <cuda_runtime_api.h>
#include <array>

typedef std::array<int, 2> intVector;
typedef std::array<double, 2> doubleVector;

template <typename T>
__host__ __device__ double operator*(const doubleVector& one, const std::array<T, 2>& two) {
    return one[0] * two[0] + one[1] * two[1];
}

template <typename T>
__host__ __device__ std::array<T, 2> operator+(const std::array<T, 2>& one, const std::array<T, 2>& two) {
    return { one[0] + two[0], one[1] + two[1] };
}

template <typename T>
__host__ __device__ std::array<T, 2> operator-(const std::array<T, 2>& one, const std::array<T, 2>& two) {
    return { one[0] - two[0], one[1] - two[1] };
}

template <typename T>
__host__ __device__ std::array<T, 2> operator*(double k, const std::array<T, 2>& v) {
    return { k * v[0], k * v[1] };
}

template <typename T>
__host__ __device__ doubleVector operator/(const std::array<T, 2>& v, double k) {
    return { v[0] / k, v[1] / k };
}

#endif //CALC_BASE2D_H
