cmake_minimum_required(VERSION 3.16)
project(nmpc_solver VERSION 1.0.0 LANGUAGES CXX)

# ── C++17 required for constexpr if, structured bindings, etc. ──
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler flags for embedded robustness ──
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -ffast-math
        -fno-exceptions          # No exceptions for embedded
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
        $<$<CONFIG:Debug>:-O0 -g -fsanitize=address,undefined>
    )
elseif(MSVC)
    add_compile_options(
        /W4 /wd4100
        /EHsc                     # Keep EH for MSVC compat, but don't throw
    )
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
endif()

# ── Header-only library ──
add_library(nmpc_solver INTERFACE)
target_include_directories(nmpc_solver INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(nmpc_solver INTERFACE cxx_std_17)

# ── Examples ──
option(NMPC_BUILD_EXAMPLES "Build example applications" ON)
if(NMPC_BUILD_EXAMPLES)
    add_executable(pendulum_nmpc_paper examples/pendulum_nmpc_paper.cpp)
    target_link_libraries(pendulum_nmpc_paper PRIVATE nmpc_solver)

    add_executable(quadrotor_2d_nmpc examples/quadrotor_2d_nmpc.cpp)
    target_link_libraries(quadrotor_2d_nmpc PRIVATE nmpc_solver)

    add_executable(chain_mass_nmpc examples/chain_mass_nmpc.cpp)
    target_link_libraries(chain_mass_nmpc PRIVATE nmpc_solver)
endif()

# ── Tests ──
option(NMPC_BUILD_TESTS "Build unit tests" ON)
if(NMPC_BUILD_TESTS)
    enable_testing()

    # Existing Riccati test
    add_executable(test_riccati tests/test_riccati.cpp)
    target_link_libraries(test_riccati PRIVATE nmpc_solver)
    add_test(NAME RiccatiRecursion COMMAND test_riccati)

    # Phase A: Filter line-search tests
    add_executable(test_filter_ls tests/test_filter_ls.cpp)
    target_link_libraries(test_filter_ls PRIVATE nmpc_solver)
    add_test(NAME FilterLS COMMAND test_filter_ls)

    # Phase B: Hessian approximation tests
    add_executable(test_hessian_approx tests/test_hessian_approx.cpp)
    target_link_libraries(test_hessian_approx PRIVATE nmpc_solver)
    add_test(NAME HessianApprox COMMAND test_hessian_approx)

    # Phase C: Barrier manager tests
    add_executable(test_barrier_manager tests/test_barrier_manager.cpp)
    target_link_libraries(test_barrier_manager PRIVATE nmpc_solver)
    add_test(NAME BarrierManager COMMAND test_barrier_manager)

    # Phase D: KKT diagnostics tests
    add_executable(test_kkt_diag tests/test_kkt_diag.cpp)
    target_link_libraries(test_kkt_diag PRIVATE nmpc_solver)
    add_test(NAME KKTDiag COMMAND test_kkt_diag)

    # Phase E: Globalization features tests
    add_executable(test_globalization tests/test_globalization.cpp)
    target_link_libraries(test_globalization PRIVATE nmpc_solver)
    add_test(NAME Globalization COMMAND test_globalization)

    # Phase F: KKT Preconditioner tests
    add_executable(test_preconditioner tests/test_preconditioner.cpp)
    target_link_libraries(test_preconditioner PRIVATE nmpc_solver)
    add_test(NAME Preconditioner COMMAND test_preconditioner)
endif()
