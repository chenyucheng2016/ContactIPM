cmake_minimum_required(VERSION 3.15)
project(impact_comparison_build LANGUAGES C CXX)

if(NOT DEFINED IMPACT_SOURCE_DIR)
    message(FATAL_ERROR "Set IMPACT_SOURCE_DIR to the downloaded IMPACT repository")
endif()
if(NOT DEFINED EIGEN_BLAS_LIBRARY OR NOT DEFINED EIGEN_LAPACK_LIBRARY)
    message(FATAL_ERROR
        "Set EIGEN_BLAS_LIBRARY and EIGEN_LAPACK_LIBRARY to the local libraries")
endif()

# CMake 3.16's FindBLAS/FindLAPACK modules do not provide the imported targets
# expected by IMPACT. Supply that compatibility layer outside the competitor
# repository so its checked-in source remains untouched.
add_library(BLAS::BLAS UNKNOWN IMPORTED)
set_target_properties(BLAS::BLAS PROPERTIES
    IMPORTED_LOCATION "${EIGEN_BLAS_LIBRARY}")
add_library(LAPACK::LAPACK UNKNOWN IMPORTED)
set_target_properties(LAPACK::LAPACK PROPERTIES
    IMPORTED_LOCATION "${EIGEN_LAPACK_LIBRARY}")
set(BLAS_LIBRARIES "${EIGEN_BLAS_LIBRARY}" CACHE STRING "" FORCE)
set(LAPACK_LIBRARIES
    "${EIGEN_LAPACK_LIBRARY};${EIGEN_BLAS_LIBRARY}" CACHE STRING "" FORCE)

# The CasADi wheel exports casadi::casadi, whereas IMPACT links the
# un-namespaced target exported by its original environment.
find_package(casadi REQUIRED CONFIG)
add_library(casadi INTERFACE)
target_link_libraries(casadi INTERFACE casadi::casadi)

set(IMPACT_BUILD_ALLEGRO OFF CACHE BOOL "" FORCE)
set(IMPACT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IMPACT_NATIVE_ARCH ON CACHE BOOL "" FORCE)

add_subdirectory("${IMPACT_SOURCE_DIR}" impact)
