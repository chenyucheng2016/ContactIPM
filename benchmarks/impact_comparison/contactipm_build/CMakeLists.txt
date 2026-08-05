cmake_minimum_required(VERSION 3.16)
project(contactipm_impact_benchmarks LANGUAGES CXX)

if(NOT DEFINED CONTACTIPM_SOURCE_DIR)
    message(FATAL_ERROR "Set CONTACTIPM_SOURCE_DIR to the ContactIPM repository")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic -Wno-unused-parameter -ffast-math
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-DNDEBUG>)
endif()

add_library(nmpc_solver INTERFACE)
target_include_directories(nmpc_solver INTERFACE
    "${CONTACTIPM_SOURCE_DIR}/include")

add_executable(contact_impact_push_box
    "${CONTACTIPM_SOURCE_DIR}/benchmarks/impact_comparison/contactipm_push_box.cpp")
target_link_libraries(contact_impact_push_box PRIVATE nmpc_solver)

add_executable(contact_impact_push_t
    "${CONTACTIPM_SOURCE_DIR}/benchmarks/impact_comparison/contactipm_push_t.cpp")
target_link_libraries(contact_impact_push_t PRIVATE nmpc_solver)

add_executable(contact_impact_cart_transport
    "${CONTACTIPM_SOURCE_DIR}/benchmarks/impact_comparison/contactipm_cart_transport.cpp")
target_link_libraries(contact_impact_cart_transport PRIVATE nmpc_solver)
