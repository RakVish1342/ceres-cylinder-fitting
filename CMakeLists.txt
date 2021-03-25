set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_minimum_required(VERSION 3.7)

#set(CMAKE_BUILD_TYPE Debug)

project(testCeres)

find_package(Ceres REQUIRED)

set(EIGEN_DIRS "/usr/include/eigen3")
#set(CERES_DIRS "/usr/local/include/ceres")

#include_directories(include ${EIGEN_DIRS} ${CERES_DIRS})
include_directories(include ${EIGEN_DIRS} ${CERES_INCLUDE_DIRS})

set(sources 
    src/cylinder_fitting.cc)

add_executable(testCeres ${sources})

target_link_libraries(testCeres ${CERES_LIBRARIES})
