cmake_minimum_required(VERSION 3.12.1)

set(PROJECT_NAME calc)
set(PROJECT_BINARY_DIR bin)

set(SOURCES src/base/ExplicitScheme.cpp src/schemes/SimpleScheme.cpp src/main.cpp src/schemes/LaxWendroffScheme.cpp)

add_executable(main ${SOURCES})
