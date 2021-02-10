cmake_minimum_required(VERSION 3.5)

project(calc
        VERSION 0.1.0
        LANGUAGES CXX CUDA)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")
set(CMAKE_CUDA_FLAGS "-std=c++14 -O3 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --expt-relaxed-constexpr -lineinfo -gencode arch=compute_75,code=sm_75")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SCHEMES_SRC src/schemes/ExplicitScheme.cpp src/schemes/SimpleScheme.cpp src/schemes/LaxWendroffScheme.cpp src/schemes/main.cpp)
add_executable(schemes ${SCHEMES_SRC})

set(LIFE_SRC src/life/main.cpp  src/life/LifeData.cpp src/life/DistributedLifeData.cpp src/life/DistributedLifeData.h)
add_executable(life ${LIFE_SRC})

set(CLIFE_SRC src/clife/main.cpp src/clife/Data.cpp src/clife/Worker.cpp src/clife/Master.cpp)
add_executable(clife ${CLIFE_SRC})

set(THERMOS_SRC src/thermos/main.cpp src/thermos/Data.cpp)
add_executable(thermos ${THERMOS_SRC})

# Including lib directories
include_directories(libs)
include_directories(libs/ranges/include)

# Range-V3
add_library(range-v3 STATIC "")
set_target_properties(range-v3 PROPERTIES LINKER_LANGUAGE CXX)
target_sources(range-v3 PRIVATE libs/ranges/include/range/v3/all.hpp)

# Dodulad CI library
add_library(dodulad_ci STATIC "")
set_target_properties(dodulad_ci PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_sources(dodulad_ci PRIVATE
        libs/dodulad_ci/ci.h
        libs/dodulad_ci/ci.cpp libs/dodulad_ci/ci_impl.hpp libs/dodulad_ci/ci_impl.cu
        libs/dodulad_ci/korobov.hpp
        libs/dodulad_ci/sse.hpp
        libs/dodulad_ci/sse.hpp libs/dodulad_ci/sse_impl.hpp
        libs/dodulad_ci/v.hpp)

add_executable(flow)
set_target_properties(flow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(flow dodulad_ci)
target_sources(flow PRIVATE
        src/flow/main.cu
        src/flow/utils/cuda.h

        src/flow/interfaces/base3D.cu
        src/flow/State3D.h src/flow/State3D.cu
        src/flow/Grid3D.h src/flow/Grid3D.cu
        src/flow/Evolution3D.h src/flow/Evolution3D.cu
        src/flow/Storage2D.h src/flow/Storage2D.cu
        src/flow/RoundHoleTank.cuh src/flow/RoundHoleTank.cu

        src/flow/DoduladCI.cu src/flow/DoduladCI.h
#        src/flow/VagapovaCI.h src/flow/VagapovaCI.h
        src/flow/interfaces/CollisionIntegral.h)

set(TEST_SRC
        src/flow/test.cpp
        src/flow/interfaces/Evolution.h src/flow/interfaces/Geometry.h
        src/flow/interfaces/CollisionIntegral.h src/flow/interfaces/Space.h
        src/flow/interfaces/Grid.h src/flow/interfaces/Storage.h)
add_executable(test ${TEST_SRC})

add_executable(cuda_test)
set_target_properties(cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_sources(cuda_test PRIVATE src/flow/test.cu)
