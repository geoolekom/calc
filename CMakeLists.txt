cmake_minimum_required(VERSION 3.5)

project(calc
        VERSION 0.1.0
        LANGUAGES CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

set(SOURCES src/base/ExplicitScheme.cpp src/schemes/SimpleScheme.cpp src/schemes/LaxWendroffScheme.cpp src/main.cpp)
add_executable(main ${SOURCES})
