cmake_minimum_required(VERSION 3.5)

project(calc
        VERSION 0.1.0
        LANGUAGES CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

set(SOURCES src/schemes/ExplicitScheme.cpp src/schemes/SimpleScheme.cpp src/schemes/LaxWendroffScheme.cpp src/schemes/main.cpp)
add_executable(main ${SOURCES})

set(LIFE_SRC src/life/main.cpp  src/life/LifeData.cpp src/life/DistributedLifeData.cpp src/life/DistributedLifeData.h)
add_executable(life ${LIFE_SRC})