cmake_minimum_required(VERSION 3.5)

project(calc
        VERSION 0.1.0
        LANGUAGES CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

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
target_sources(dodulad_ci PRIVATE
        libs/dodulad_ci/ci.hpp
        libs/dodulad_ci/ci.cpp libs/dodulad_ci/ci_impl.hpp
        libs/dodulad_ci/korobov.hpp
        libs/dodulad_ci/sse.hpp libs/dodulad_ci/sse.hpp libs/dodulad_ci/sse_impl.hpp libs/dodulad_ci/v.hpp)

add_executable(flow)
target_link_libraries(flow dodulad_ci range-v3)
target_sources(flow PRIVATE src/flow/main.cpp
        src/flow/interfaces/base.h
        src/flow/State2D.cpp src/flow/State2D.h
        src/flow/Grid2D.cpp src/flow/Grid2D.h
        src/flow/Evolution2D.cpp src/flow/Evolution2D.h
        src/flow/Tank2D.cpp src/flow/Tank2D.h
        src/flow/TankWithScreen2D.cpp src/flow/TankWithScreen2D.h
        src/flow/IndexTankWithScreen2D.cpp src/flow/IndexTankWithScreen2D.h
        src/flow/Storage2D.cpp src/flow/Storage2D.h
        src/flow/IndexMap.cpp src/flow/IndexMap.h
        src/flow/DoduladCI.cpp src/flow/DoduladCI.h
        src/flow/VagapovaCI.h src/flow/VagapovaCI.h
        src/flow/interfaces/CollisionIntegral.h)

set(TEST_SRC
        src/flow/test.cpp
        src/flow/interfaces/Evolution.h src/flow/interfaces/Geometry.h
        src/flow/interfaces/CollisionIntegral.h src/flow/interfaces/Space.h
        src/flow/interfaces/Grid.h src/flow/interfaces/Storage.h)
add_executable(test ${TEST_SRC})
