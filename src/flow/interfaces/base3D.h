#ifndef CALC_BASE3D_H
#define CALC_BASE3D_H

#include <host_defines.h>
#include <array>

typedef std::array<int, 3> intVector;
typedef std::array<double, 3> doubleVector;

template <typename T>
__host__ __device__ double operator*(const doubleVector& one, const std::array<T, 3>& two) {
    return one[0] * two[0] + one[1] * two[1] + one[2] * two[2];
}

template <typename T>
__host__ __device__ std::array<T, 3> operator+(const std::array<T, 3>& one, const std::array<T, 3>& two) {
    return { one[0] + two[0], one[1] + two[1], one[2] + two[2] };
}

template <typename T>
__host__ __device__ std::array<T, 3> operator-(const std::array<T, 3>& one, const std::array<T, 3>& two) {
    return { one[0] - two[0], one[1] - two[1], one[2] - two[2] };
}

template <typename T, typename M>
__host__ __device__ std::array<T, 3> operator*(M k, const std::array<T, 3>& v) {
    return { k * v[0], k * v[1], k * v[2] };
}

template <typename T>
__host__ __device__ doubleVector operator/(const std::array<T, 3>& v, double k) {
    return { v[0] / k, v[1] / k, v[2] / k };
}

#endif //CALC_BASE3D_H