CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
PROJECT(pydusa
		LANGUAGES C
		)

include(CMakePrintHelpers)

include("functions.cmake")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

find_package(Conda REQUIRED)
#include("FindPython.cmake")
find_package(Python  REQUIRED)
find_package(NumPy  REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(MPI REQUIRED)

cmake_print_variables(MPI_C_COMPILER)


add_library(mpi SHARED src/mympimodule.c)
target_include_directories(mpi PRIVATE
        ${PYTHON_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIR}
        ${FFTW3D_INCLUDE_PATH}
        )

target_link_libraries(mpi MPI::MPI_C)

install(TARGETS mpi DESTINATION ${SP_DIR})
