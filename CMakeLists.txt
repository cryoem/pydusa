cmake_minimum_required(VERSION 3.5)
project(pydusa_1_15_sparx_8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp src/temp.c)
add_executable(pydusa_1_15_sparx_8 ${SOURCE_FILES})