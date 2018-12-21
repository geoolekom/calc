cmake_minimum_required(VERSION 3.5)

project(calc
        VERSION 0.1.0
        LANGUAGES CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
set(CMAKE_CXX_FLAGS "-std=c++11")

set(SCHEMES_SRC src/schemes/ExplicitScheme.cpp src/schemes/SimpleScheme.cpp src/schemes/LaxWendroffScheme.cpp src/schemes/main.cpp)
add_executable(schemes ${SCHEMES_SRC})

set(LIFE_SRC src/life/main.cpp  src/life/LifeData.cpp src/life/DistributedLifeData.cpp src/life/DistributedLifeData.h)
add_executable(life ${LIFE_SRC})

set(CLIFE_SRC src/clife/main.cpp src/clife/Data.cpp src/clife/Worker.cpp src/clife/Master.cpp)
add_executable(clife ${CLIFE_SRC})

set(THERMOS_SRC src/thermos/main.cpp src/thermos/Data.cpp)
add_executable(thermos ${THERMOS_SRC})

set(FLOW_SRC src/flow/main.cpp src/flow/State2D.cpp src/flow/State2D.h src/flow/Grid2D.cpp src/flow/Grid2D.h src/flow/Evolution2D.cpp src/flow/Evolution2D.h src/flow/Tank2D.cpp src/flow/Tank2D.h)
add_executable(flow ${FLOW_SRC})
