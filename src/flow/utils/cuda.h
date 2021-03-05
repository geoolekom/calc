//
// Created by geoolekom on 27.04.19.
//

#ifndef CALC_CUDA_H
#define CALC_CUDA_H

#include <cuda_runtime_api.h>
#include <iostream>
#include <typeinfo>

template <typename T, typename... Args> void cudaNew(T **ptr, Args... args) {
    auto temp = T(args...);
    cudaMallocManaged((void **)ptr, sizeof(T), cudaMemAttachGlobal);
    cudaMemcpy(*ptr, &temp, sizeof(T), cudaMemcpyHostToDevice);
    auto ret = cudaDeviceSynchronize();
    if (ret != 0) {
        std::cout << "Выделение видеопамяти. Ошибка: " << typeid(T).name() << ", " << cudaGetErrorString(ret)
                  << std::endl;
    }
}

template <typename T> void cudaCopy(T **ptr, T *data) {
    cudaMallocManaged((void **)ptr, sizeof(T), cudaMemAttachGlobal);
    cudaMemcpy(*ptr, data, sizeof(T), cudaMemcpyHostToDevice);
    auto ret = cudaDeviceSynchronize();
    if (ret != 0) {
        std::cout << "Выделение видеопамяти. Ошибка: " << typeid(T).name() << ", " << cudaGetErrorString(ret)
                  << std::endl;
    }
}

#endif // CALC_CUDA_H
