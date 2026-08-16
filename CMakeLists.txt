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
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-DNDEBUG>
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
elseif(MSVC)
    add_compile_options(
        /W4 /wd4100
        /EHsc                     # Keep EH for MSVC compat, but don't throw
    )
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    # Increase stack size to 10MB for large NMPC problems (default is 1MB)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
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

    add_executable(test_lqr_invariance examples/test_lqr_invariance.cpp)
    target_link_libraries(test_lqr_invariance PRIVATE nmpc_solver)

    add_executable(chain_mass_nmpc examples/chain_mass_nmpc.cpp)
    target_link_libraries(chain_mass_nmpc PRIVATE nmpc_solver)

    add_executable(centroidal_trot_nmpc examples/centroidal_trot_nmpc.cpp)
    target_link_libraries(centroidal_trot_nmpc PRIVATE nmpc_solver)

    add_executable(quadruped_cito_realtime_benchmark
        examples/quadruped_cito/quadruped_cito_realtime_benchmark.cpp)
    target_include_directories(quadruped_cito_realtime_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(quadruped_cito_realtime_benchmark PRIVATE
        nmpc_solver)

    add_executable(quadruped_cito_srbd_closed_loop
        examples/quadruped_cito/quadruped_cito_srbd_closed_loop.cpp)
    target_include_directories(quadruped_cito_srbd_closed_loop PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(quadruped_cito_srbd_closed_loop PRIVATE
        nmpc_solver)
endif()

# ── Tests ──
option(NMPC_BUILD_MUJOCO "Build the optional MuJoCo quadruped bridge" OFF)
if(NMPC_BUILD_MUJOCO)
    find_package(Threads REQUIRED)
    set(MUJOCO_ROOT "" CACHE PATH
        "Directory containing include/mujoco/mujoco.h and libmujoco")
    set(MUJOCO_MENAGERIE_ROOT "" CACHE PATH
        "MuJoCo Menagerie checkout containing unitree_go1/scene.xml")
    find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
        HINTS "${MUJOCO_ROOT}/include"
        NO_DEFAULT_PATH)
    find_file(MUJOCO_LIBRARY
        NAMES libmujoco.so.3.9.0 libmujoco.so mujoco.dll
        HINTS "${MUJOCO_ROOT}"
        NO_DEFAULT_PATH)
    if(NOT MUJOCO_INCLUDE_DIR OR NOT MUJOCO_LIBRARY)
        message(FATAL_ERROR
            "NMPC_BUILD_MUJOCO requires MUJOCO_ROOT with MuJoCo headers "
            "and the native library")
    endif()
    if(NOT EXISTS "${MUJOCO_MENAGERIE_ROOT}/unitree_go1/scene.xml")
        message(FATAL_ERROR
            "MUJOCO_MENAGERIE_ROOT must contain unitree_go1/scene.xml")
    endif()
    set(MUJOCO_GO1_XML
        "${MUJOCO_MENAGERIE_ROOT}/unitree_go1/go1.xml")
    set(QUADRUPED_CITO_MUJOCO_SCENE
        "${CMAKE_CURRENT_BINARY_DIR}/go1_cito_terrain_scene.xml")
    configure_file(
        examples/quadruped_cito/mujoco/go1_terrain_scene.xml.in
        "${QUADRUPED_CITO_MUJOCO_SCENE}" @ONLY)

    add_library(mujoco_external SHARED IMPORTED)
    set_target_properties(mujoco_external PROPERTIES
        IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}")
    get_filename_component(MUJOCO_LIBRARY_DIR "${MUJOCO_LIBRARY}" DIRECTORY)

    add_library(quadruped_cito_mujoco_adapter STATIC
        examples/quadruped_cito/mujoco/mujoco_go1_adapter.cpp)
    target_include_directories(quadruped_cito_mujoco_adapter PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(quadruped_cito_mujoco_adapter
        PUBLIC nmpc_solver mujoco_external)

    add_executable(quadruped_cito_mujoco
        examples/quadruped_cito/mujoco/quadruped_cito_mujoco.cpp)
    target_include_directories(quadruped_cito_mujoco PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(quadruped_cito_mujoco PRIVATE
        quadruped_cito_mujoco_adapter Threads::Threads)
    set_target_properties(quadruped_cito_mujoco PROPERTIES
        BUILD_RPATH "${MUJOCO_LIBRARY_DIR}")
endif()

option(NMPC_BUILD_TESTS "Build unit tests" ON)
if(NMPC_BUILD_TESTS)
    enable_testing()

    # Existing Riccati test
    add_executable(test_riccati tests/test_riccati.cpp)
    target_link_libraries(test_riccati PRIVATE nmpc_solver)
    add_test(NAME RiccatiRecursion COMMAND test_riccati)

    add_executable(test_riccati_equivalence
        tests/test_riccati_equivalence.cpp)
    target_link_libraries(test_riccati_equivalence PRIVATE nmpc_solver)
    add_test(NAME RiccatiEquivalence COMMAND test_riccati_equivalence)

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

    # Preconditioner invariance regression test
    add_executable(test_precond_invariance tests/test_precond_invariance.cpp)
    target_link_libraries(test_precond_invariance PRIVATE nmpc_solver)
    add_test(NAME PrecondInvariance COMMAND test_precond_invariance)
    # MPCC recognition and Eq. 3.13 elastic slack/barrier regression
    add_executable(test_mpcc_relaxation tests/test_mpcc_relaxation.cpp)
    target_link_libraries(test_mpcc_relaxation PRIVATE nmpc_solver)
    add_test(NAME MPCCRelaxation COMMAND test_mpcc_relaxation)

    add_executable(test_quadruped_cito_model
        tests/test_quadruped_cito_model.cpp)
    target_include_directories(test_quadruped_cito_model PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_quadruped_cito_model PRIVATE nmpc_solver)
    add_test(NAME QuadrupedCITOModel COMMAND test_quadruped_cito_model)

    add_executable(test_quadruped_cito_realtime
        tests/test_quadruped_cito_realtime.cpp)
    target_include_directories(test_quadruped_cito_realtime PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_quadruped_cito_realtime PRIVATE nmpc_solver)
    add_test(NAME QuadrupedCITORealtimeInvariants
        COMMAND test_quadruped_cito_realtime)

    add_executable(test_quadruped_cito_realtime_execution
        tests/test_quadruped_cito_realtime_execution.cpp)
    target_include_directories(test_quadruped_cito_realtime_execution PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME QuadrupedCITORealtimeExecutionTiming
        COMMAND test_quadruped_cito_realtime_execution)

    add_executable(test_quadruped_cito_rolling_audit
        tests/test_quadruped_cito_rolling_audit.cpp)
    target_include_directories(test_quadruped_cito_rolling_audit PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_quadruped_cito_rolling_audit PRIVATE nmpc_solver)
    add_test(NAME QuadrupedCITORollingAudit
        COMMAND test_quadruped_cito_rolling_audit)

    add_executable(test_quadruped_cito_sustained_topology
        tests/test_quadruped_cito_sustained_topology.cpp)
    target_include_directories(test_quadruped_cito_sustained_topology PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_quadruped_cito_sustained_topology PRIVATE
        nmpc_solver)
    add_test(NAME QuadrupedCITOSustainedTopology
        COMMAND test_quadruped_cito_sustained_topology)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME QuadrupedCITORealtimeCampaignProtocol
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_realtime_campaign_protocol.py)
        add_test(NAME QuadrupedCITOSustainedRealtimeCampaignProtocol
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sustained_realtime_campaign.py)
    endif()

    add_executable(test_quadruped_cito_solve
        tests/test_quadruped_cito_solve.cpp)
    target_include_directories(test_quadruped_cito_solve PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_quadruped_cito_solve PRIVATE nmpc_solver)
    add_test(NAME QuadrupedCITOSolve COMMAND test_quadruped_cito_solve)

    function(add_quadruped_cito_transition_test target test_name horizon dt)
        add_executable(${target} tests/test_quadruped_cito_transition.cpp)
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(${target} PRIVATE
            QUADRUPED_CITO_TEST_HORIZON=${horizon}
            QUADRUPED_CITO_TEST_TIMESTEP=${dt})
        target_link_libraries(${target} PRIVATE nmpc_solver)
        add_test(NAME ${test_name} COMMAND ${target})
    endfunction()

    add_quadruped_cito_transition_test(
        test_quadruped_cito_transition QuadrupedCITOTransition20 20 0.08)
    add_quadruped_cito_transition_test(
        test_quadruped_cito_transition_40 QuadrupedCITOTransition40 40 0.04)
    add_quadruped_cito_transition_test(
        test_quadruped_cito_transition_60 QuadrupedCITOTransition60 60
        0.0266666666666667)
    add_quadruped_cito_transition_test(
        test_quadruped_cito_transition_80 QuadrupedCITOTransition80 80 0.02)

    add_quadruped_cito_transition_test(
        test_quadruped_cito_terrain_smooth6 QuadrupedCITOTerrainSmooth6
        20 0.08)
    target_compile_definitions(test_quadruped_cito_terrain_smooth6 PRIVATE
        QUADRUPED_CITO_TEST_TERRAIN_AMPLITUDE=0.06)

    add_quadruped_cito_transition_test(
        test_quadruped_cito_terrain_cross20 QuadrupedCITOTerrainCrossSlope20
        20 0.08)
    target_compile_definitions(test_quadruped_cito_terrain_cross20 PRIVATE
        QUADRUPED_CITO_TEST_TERRAIN_AMPLITUDE=0.0
        QUADRUPED_CITO_TEST_TERRAIN_SLOPE_Y=0.3639702342662023)

    add_quadruped_cito_transition_test(
        test_quadruped_cito_terrain_step4 QuadrupedCITOTerrainStep4
        20 0.08)
    target_compile_definitions(test_quadruped_cito_terrain_step4 PRIVATE
        QUADRUPED_CITO_TEST_SMOOTH_STEP=1
        QUADRUPED_CITO_TEST_USE_RECOVERY=1)

    add_quadruped_cito_transition_test(
        test_quadruped_cito_go1_flat QuadrupedCITOGo1Flat 20 0.08)
    target_compile_definitions(test_quadruped_cito_go1_flat PRIVATE
        QUADRUPED_CITO_TEST_GO1=1
        QUADRUPED_CITO_TEST_USE_RECOVERY=1)
    add_quadruped_cito_transition_test(
        test_quadruped_cito_go1_smooth QuadrupedCITOGo1Smooth 20 0.08)
    target_compile_definitions(test_quadruped_cito_go1_smooth PRIVATE
        QUADRUPED_CITO_TEST_GO1=1
        QUADRUPED_CITO_TEST_GO1_SMOOTH=1
        QUADRUPED_CITO_TEST_USE_RECOVERY=1)
    add_quadruped_cito_transition_test(
        test_quadruped_cito_go1_slope QuadrupedCITOGo1Slope 20 0.08)
    target_compile_definitions(test_quadruped_cito_go1_slope PRIVATE
        QUADRUPED_CITO_TEST_GO1=1
        QUADRUPED_CITO_TEST_GO1_SLOPE=1
        QUADRUPED_CITO_TEST_USE_RECOVERY=1)

    add_executable(test_quadruped_cito_multifoot
        tests/test_quadruped_cito_multifoot.cpp)
    target_include_directories(test_quadruped_cito_multifoot PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_quadruped_cito_multifoot PRIVATE nmpc_solver)
    add_test(NAME QuadrupedCITOMultifoot
        COMMAND test_quadruped_cito_multifoot)

    add_executable(test_quadruped_cito_long_horizon
        tests/test_quadruped_cito_multifoot.cpp)
    target_include_directories(test_quadruped_cito_long_horizon PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(test_quadruped_cito_long_horizon PRIVATE
        QUADRUPED_CITO_LONG_HORIZON=1)
    target_link_libraries(test_quadruped_cito_long_horizon PRIVATE nmpc_solver)
    add_test(NAME QuadrupedCITOLongHorizon
        COMMAND test_quadruped_cito_long_horizon)

    add_executable(test_quadruped_cito_four_step
        tests/test_quadruped_cito_four_step_planner.cpp)
    target_include_directories(test_quadruped_cito_four_step PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_quadruped_cito_four_step PRIVATE nmpc_solver)
    add_test(NAME QuadrupedCITOFourStep
        COMMAND test_quadruped_cito_four_step)
    add_test(NAME QuadrupedCITOFourStepMultistart
        COMMAND test_quadruped_cito_four_step --multistart)
    add_test(NAME QuadrupedCITOFourStepRandomSmooth
        COMMAND test_quadruped_cito_four_step --terrain random_smooth)

    add_executable(test_quadruped_cito_eight_step
        tests/test_quadruped_cito_multifoot.cpp)
    target_include_directories(test_quadruped_cito_eight_step PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(test_quadruped_cito_eight_step PRIVATE
        QUADRUPED_CITO_LONG_HORIZON=1
        QUADRUPED_CITO_EIGHT_STEP=1)
    target_link_libraries(test_quadruped_cito_eight_step PRIVATE nmpc_solver)

    add_executable(test_quadruped_cito_execution
        tests/test_quadruped_cito_execution.cpp)
    target_include_directories(test_quadruped_cito_execution PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_quadruped_cito_execution PRIVATE nmpc_solver)
    add_test(NAME QuadrupedCITOExecution
        COMMAND test_quadruped_cito_execution)

    if(NMPC_BUILD_MUJOCO)
        add_executable(test_quadruped_cito_mujoco_adapter
            tests/test_quadruped_cito_mujoco_adapter.cpp)
        target_include_directories(test_quadruped_cito_mujoco_adapter PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(test_quadruped_cito_mujoco_adapter PRIVATE
            quadruped_cito_mujoco_adapter)
        set_target_properties(test_quadruped_cito_mujoco_adapter PROPERTIES
            BUILD_RPATH "${MUJOCO_LIBRARY_DIR}")
        add_test(NAME QuadrupedCITOMujocoAdapter
            COMMAND test_quadruped_cito_mujoco_adapter
                    "${QUADRUPED_CITO_MUJOCO_SCENE}")
        add_executable(test_quadruped_cito_mujoco_snapshot
            tests/test_quadruped_cito_mujoco_snapshot.cpp)
        target_include_directories(test_quadruped_cito_mujoco_snapshot PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(test_quadruped_cito_mujoco_snapshot PRIVATE
            quadruped_cito_mujoco_adapter)
        set_target_properties(test_quadruped_cito_mujoco_snapshot PROPERTIES
            BUILD_RPATH "${MUJOCO_LIBRARY_DIR}")
        add_test(NAME QuadrupedCITOMujocoSnapshot
            COMMAND test_quadruped_cito_mujoco_snapshot
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    "${CMAKE_CURRENT_BINARY_DIR}/quadruped_cito_snapshot.txt")
        add_test(NAME QuadrupedCITOMujocoStanding
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --stand 60)
        set(QUADRUPED_CITO_MUJOCO_PLAN
            "${CMAKE_CURRENT_BINARY_DIR}/quadruped_cito_transition.plan")
        add_test(NAME QuadrupedCITOMujocoPlanExport
            COMMAND test_quadruped_cito_go1_flat
                    --write-plan "${QUADRUPED_CITO_MUJOCO_PLAN}")
        add_test(NAME QuadrupedCITOMujocoSingleFootReplay
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain flat
                    --replay-plan "${QUADRUPED_CITO_MUJOCO_PLAN}")

        set(QUADRUPED_CITO_MUJOCO_SMOOTH_PLAN
            "${CMAKE_CURRENT_BINARY_DIR}/quadruped_cito_go1_smooth.plan")
        add_test(NAME QuadrupedCITOMujocoSmoothPlanExport
            COMMAND test_quadruped_cito_go1_smooth
                    --write-plan "${QUADRUPED_CITO_MUJOCO_SMOOTH_PLAN}")
        add_test(NAME QuadrupedCITOMujocoSmoothSingleFootReplay
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain smooth
                    --replay-plan "${QUADRUPED_CITO_MUJOCO_SMOOTH_PLAN}")

        set(QUADRUPED_CITO_MUJOCO_SLOPE_PLAN
            "${CMAKE_CURRENT_BINARY_DIR}/quadruped_cito_go1_slope.plan")
        add_test(NAME QuadrupedCITOMujocoSlopePlanExport
            COMMAND test_quadruped_cito_go1_slope
                    --write-plan "${QUADRUPED_CITO_MUJOCO_SLOPE_PLAN}")
        add_test(NAME QuadrupedCITOMujocoSlopeSingleFootReplay
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain slope
                    --replay-plan "${QUADRUPED_CITO_MUJOCO_SLOPE_PLAN}")
        set(QUADRUPED_CITO_MUJOCO_MULTIFOOT_FORWARD_PLAN
            "${CMAKE_CURRENT_BINARY_DIR}/quadruped_cito_multifoot_forward.plan")
        set(QUADRUPED_CITO_MUJOCO_MULTIFOOT_REVERSE_PLAN
            "${CMAKE_CURRENT_BINARY_DIR}/quadruped_cito_multifoot_reverse.plan")
        add_test(NAME QuadrupedCITOMujocoMultifootPlanExport
            COMMAND test_quadruped_cito_multifoot
                    --write-forward
                    "${QUADRUPED_CITO_MUJOCO_MULTIFOOT_FORWARD_PLAN}"
                    --write-reverse
                    "${QUADRUPED_CITO_MUJOCO_MULTIFOOT_REVERSE_PLAN}")
        add_test(NAME QuadrupedCITOMujocoMultifootForwardReplay
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain flat
                    --replay-plan
                    "${QUADRUPED_CITO_MUJOCO_MULTIFOOT_FORWARD_PLAN}")
        add_test(NAME QuadrupedCITOMujocoMultifootReverseReplay
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain flat
                    --replay-plan
                    "${QUADRUPED_CITO_MUJOCO_MULTIFOOT_REVERSE_PLAN}")
        add_test(NAME QuadrupedCITOMujocoCombinedRobustnessReplay
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain flat
                    --replay-plan
                    "${QUADRUPED_CITO_MUJOCO_MULTIFOOT_FORWARD_PLAN}"
                    --mass-scale 1.10
                    --friction-scale 0.75
                    --push 0 20 0 0.90 0.10)
        add_test(NAME QuadrupedCITOMujocoUniformForceBaseline
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain flat
                    --replay-plan
                    "${QUADRUPED_CITO_MUJOCO_MULTIFOOT_FORWARD_PLAN}"
                    --force-reference uniform)
        set(QUADRUPED_CITO_MUJOCO_LONG_PLAN
            "${CMAKE_CURRENT_BINARY_DIR}/quadruped_cito_long_horizon.plan")
        add_test(NAME QuadrupedCITOMujocoLongPlanExport
            COMMAND test_quadruped_cito_long_horizon
                    --write-plan "${QUADRUPED_CITO_MUJOCO_LONG_PLAN}")
        add_test(NAME QuadrupedCITOMujocoLongReplay
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain flat
                    --replay-plan "${QUADRUPED_CITO_MUJOCO_LONG_PLAN}")
        set(QUADRUPED_CITO_MUJOCO_FOUR_STEP_FLAT_PLAN
            "${CMAKE_CURRENT_BINARY_DIR}/quadruped_cito_four_step_flat.plan")
        set(QUADRUPED_CITO_MUJOCO_FOUR_STEP_SMOOTH_PLAN
            "${CMAKE_CURRENT_BINARY_DIR}/quadruped_cito_four_step_smooth.plan")
        add_test(NAME QuadrupedCITOMujocoFourStepFlatPlanExport
            COMMAND test_quadruped_cito_four_step
                    --terrain flat --write-plan
                    "${QUADRUPED_CITO_MUJOCO_FOUR_STEP_FLAT_PLAN}")
        add_test(NAME QuadrupedCITOMujocoFourStepFlatReplay
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain flat --replay-plan
                    "${QUADRUPED_CITO_MUJOCO_FOUR_STEP_FLAT_PLAN}")
        add_test(NAME QuadrupedCITOMujocoFourStepSmoothPlanExport
            COMMAND test_quadruped_cito_four_step
                    --terrain smooth --write-plan
                    "${QUADRUPED_CITO_MUJOCO_FOUR_STEP_SMOOTH_PLAN}")
        add_test(NAME QuadrupedCITOMujocoFourStepSmoothReplay
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain smooth --replay-plan
                    "${QUADRUPED_CITO_MUJOCO_FOUR_STEP_SMOOTH_PLAN}")
        add_test(NAME QuadrupedCITOMujocoReplannedFlat
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain flat --replan-twice)
        add_test(NAME QuadrupedCITOMujocoReplannedSmooth
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain smooth --replan-count 2
                    --seed-policy multistart)
        add_test(NAME QuadrupedCITOMujocoReplannedFlatFourSegment
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain flat --replan-count 4 --seed-policy multistart
                    --allow-timing-miss)
        add_test(NAME QuadrupedCITOMujocoReplannedRandomSmooth
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain random_smooth --terrain-seed 7
                    --terrain-amplitude 0.02 --replan-count 2
                    --seed-policy multistart)
        set(QUADRUPED_CITO_MUJOCO_LONG_SMOOTH_FORWARD_PLAN
            "${CMAKE_CURRENT_BINARY_DIR}/quadruped_cito_long_smooth_forward.plan")
        set(QUADRUPED_CITO_MUJOCO_LONG_SMOOTH_REVERSE_PLAN
            "${CMAKE_CURRENT_BINARY_DIR}/quadruped_cito_long_smooth_reverse.plan")
        set(QUADRUPED_CITO_MUJOCO_LONG_SMOOTH_FIXED_FORWARD_PLAN
            "${CMAKE_CURRENT_BINARY_DIR}/quadruped_cito_long_smooth_fixed_forward.plan")
        set(QUADRUPED_CITO_MUJOCO_LONG_SMOOTH_FIXED_REVERSE_PLAN
            "${CMAKE_CURRENT_BINARY_DIR}/quadruped_cito_long_smooth_fixed_reverse.plan")
        add_test(NAME QuadrupedCITOMujocoLongSmoothForwardPlanExport
            COMMAND test_quadruped_cito_long_horizon
                    --terrain smooth
                    --write-plan
                    "${QUADRUPED_CITO_MUJOCO_LONG_SMOOTH_FORWARD_PLAN}"
                    --write-fixed-plan
                    "${QUADRUPED_CITO_MUJOCO_LONG_SMOOTH_FIXED_FORWARD_PLAN}")
        add_test(NAME QuadrupedCITOMujocoLongSmoothReversePlanExport
            COMMAND test_quadruped_cito_long_horizon
                    --reverse --terrain smooth
                    --write-plan
                    "${QUADRUPED_CITO_MUJOCO_LONG_SMOOTH_REVERSE_PLAN}"
                    --write-fixed-plan
                    "${QUADRUPED_CITO_MUJOCO_LONG_SMOOTH_FIXED_REVERSE_PLAN}")
        add_test(NAME QuadrupedCITOMujocoLongSmoothForwardReplay
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain smooth --replay-plan
                    "${QUADRUPED_CITO_MUJOCO_LONG_SMOOTH_FORWARD_PLAN}")
        add_test(NAME QuadrupedCITOMujocoLongSmoothReverseReplay
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain smooth --replay-plan
                    "${QUADRUPED_CITO_MUJOCO_LONG_SMOOTH_REVERSE_PLAN}")
        add_test(NAME QuadrupedCITOMujocoLongSmoothFixedForwardBaseline
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain smooth --replay-plan
                    "${QUADRUPED_CITO_MUJOCO_LONG_SMOOTH_FIXED_FORWARD_PLAN}")
        add_test(NAME QuadrupedCITOMujocoLongSmoothFixedReverseBaseline
            COMMAND quadruped_cito_mujoco
                    "${QUADRUPED_CITO_MUJOCO_SCENE}"
                    --terrain smooth --replay-plan
                    "${QUADRUPED_CITO_MUJOCO_LONG_SMOOTH_FIXED_REVERSE_PLAN}")
        set_tests_properties(
            QuadrupedCITOMujocoAdapter QuadrupedCITOMujocoStanding
            QuadrupedCITOMujocoPlanExport
            QuadrupedCITOMujocoSingleFootReplay
            QuadrupedCITOMujocoSmoothPlanExport
            QuadrupedCITOMujocoSmoothSingleFootReplay
            QuadrupedCITOMujocoSlopePlanExport
            QuadrupedCITOMujocoSlopeSingleFootReplay
            QuadrupedCITOMujocoMultifootPlanExport
            QuadrupedCITOMujocoMultifootForwardReplay
            QuadrupedCITOMujocoMultifootReverseReplay
            QuadrupedCITOMujocoCombinedRobustnessReplay
            QuadrupedCITOMujocoUniformForceBaseline
            QuadrupedCITOMujocoLongPlanExport
            QuadrupedCITOMujocoLongReplay
            QuadrupedCITOMujocoFourStepFlatPlanExport
            QuadrupedCITOMujocoFourStepFlatReplay
            QuadrupedCITOMujocoFourStepSmoothPlanExport
            QuadrupedCITOMujocoFourStepSmoothReplay
            QuadrupedCITOMujocoReplannedFlat
            QuadrupedCITOMujocoReplannedSmooth
            QuadrupedCITOMujocoReplannedFlatFourSegment
            QuadrupedCITOMujocoReplannedRandomSmooth
            QuadrupedCITOMujocoLongSmoothForwardPlanExport
            QuadrupedCITOMujocoLongSmoothReversePlanExport
            QuadrupedCITOMujocoLongSmoothForwardReplay
            QuadrupedCITOMujocoLongSmoothReverseReplay
            QuadrupedCITOMujocoLongSmoothFixedForwardBaseline
            QuadrupedCITOMujocoLongSmoothFixedReverseBaseline PROPERTIES
            LABELS "quadruped_cito;quadruped_cito_mujoco")
        set_tests_properties(QuadrupedCITOMujocoPlanExport PROPERTIES
            FIXTURES_SETUP quadruped_cito_mujoco_plan)
        set_tests_properties(QuadrupedCITOMujocoSingleFootReplay PROPERTIES
            FIXTURES_REQUIRED quadruped_cito_mujoco_plan)
        set_tests_properties(QuadrupedCITOMujocoSmoothPlanExport PROPERTIES
            FIXTURES_SETUP quadruped_cito_mujoco_smooth_plan TIMEOUT 180)
        set_tests_properties(
            QuadrupedCITOMujocoSmoothSingleFootReplay PROPERTIES
            FIXTURES_REQUIRED quadruped_cito_mujoco_smooth_plan)
        set_tests_properties(QuadrupedCITOMujocoSlopePlanExport PROPERTIES
            FIXTURES_SETUP quadruped_cito_mujoco_slope_plan TIMEOUT 180)
        set_tests_properties(
            QuadrupedCITOMujocoSlopeSingleFootReplay PROPERTIES
            FIXTURES_REQUIRED quadruped_cito_mujoco_slope_plan)
        set_tests_properties(
            QuadrupedCITOMujocoMultifootPlanExport PROPERTIES
            FIXTURES_SETUP quadruped_cito_mujoco_multifoot_plan TIMEOUT 180)
        set_tests_properties(
            QuadrupedCITOMujocoMultifootForwardReplay
            QuadrupedCITOMujocoMultifootReverseReplay
            QuadrupedCITOMujocoCombinedRobustnessReplay
            QuadrupedCITOMujocoUniformForceBaseline PROPERTIES
            FIXTURES_REQUIRED quadruped_cito_mujoco_multifoot_plan)
        set_tests_properties(
            QuadrupedCITOMujocoUniformForceBaseline PROPERTIES WILL_FAIL TRUE)
        set_tests_properties(QuadrupedCITOMujocoLongPlanExport PROPERTIES
            FIXTURES_SETUP quadruped_cito_mujoco_long_plan TIMEOUT 240)
        set_tests_properties(QuadrupedCITOMujocoLongReplay PROPERTIES
            FIXTURES_REQUIRED quadruped_cito_mujoco_long_plan)
        set_tests_properties(
            QuadrupedCITOMujocoFourStepFlatPlanExport PROPERTIES
            FIXTURES_SETUP quadruped_cito_mujoco_four_step_flat_plan
            TIMEOUT 240)
        set_tests_properties(
            QuadrupedCITOMujocoFourStepFlatReplay PROPERTIES
            FIXTURES_REQUIRED quadruped_cito_mujoco_four_step_flat_plan)
        set_tests_properties(
            QuadrupedCITOMujocoFourStepSmoothPlanExport PROPERTIES
            FIXTURES_SETUP quadruped_cito_mujoco_four_step_smooth_plan
            TIMEOUT 240)
        set_tests_properties(
            QuadrupedCITOMujocoFourStepSmoothReplay PROPERTIES
            FIXTURES_REQUIRED quadruped_cito_mujoco_four_step_smooth_plan)
        set_tests_properties(
            QuadrupedCITOMujocoLongSmoothForwardPlanExport PROPERTIES
            FIXTURES_SETUP quadruped_cito_mujoco_long_smooth_forward_plan
            TIMEOUT 240)
        set_tests_properties(
            QuadrupedCITOMujocoLongSmoothReversePlanExport PROPERTIES
            FIXTURES_SETUP quadruped_cito_mujoco_long_smooth_reverse_plan
            TIMEOUT 240)
        set_tests_properties(
            QuadrupedCITOMujocoLongSmoothForwardReplay
            QuadrupedCITOMujocoLongSmoothFixedForwardBaseline PROPERTIES
            FIXTURES_REQUIRED quadruped_cito_mujoco_long_smooth_forward_plan)
        set_tests_properties(
            QuadrupedCITOMujocoLongSmoothReverseReplay
            QuadrupedCITOMujocoLongSmoothFixedReverseBaseline PROPERTIES
            FIXTURES_REQUIRED quadruped_cito_mujoco_long_smooth_reverse_plan)
        set_tests_properties(QuadrupedCITOMujocoStanding PROPERTIES
            RUN_SERIAL TRUE TIMEOUT 120)
        set_tests_properties(
            QuadrupedCITOMujocoReplannedFlat
            QuadrupedCITOMujocoReplannedSmooth
            QuadrupedCITOMujocoReplannedFlatFourSegment
            QuadrupedCITOMujocoReplannedRandomSmooth PROPERTIES
            RUN_SERIAL TRUE TIMEOUT 300)
        set_tests_properties(
            QuadrupedCITOMujocoReplannedFlatFourSegment PROPERTIES
            TIMEOUT 360)
    endif()

    set_tests_properties(
        QuadrupedCITOModel QuadrupedCITOSolve
        QuadrupedCITOTransition20 QuadrupedCITOTransition40
        QuadrupedCITOTransition60 QuadrupedCITOTransition80
        QuadrupedCITOTerrainSmooth6 QuadrupedCITOTerrainCrossSlope20
        QuadrupedCITOTerrainStep4 QuadrupedCITOMultifoot
        QuadrupedCITOLongHorizon QuadrupedCITOFourStep
        QuadrupedCITOFourStepMultistart QuadrupedCITOFourStepRandomSmooth
        QuadrupedCITOExecution
        PROPERTIES LABELS quadruped_cito)
    set_tests_properties(
        QuadrupedCITOFourStepMultistart QuadrupedCITOFourStepRandomSmooth
        PROPERTIES RUN_SERIAL TRUE TIMEOUT 240)
    set_tests_properties(
        QuadrupedCITOTransition80 QuadrupedCITOTerrainStep4
        QuadrupedCITOMultifoot QuadrupedCITOLongHorizon
        PROPERTIES LABELS "quadruped_cito;quadruped_cito_long"
                   RUN_SERIAL TRUE TIMEOUT 240)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME ContactBenchmarkProtocol
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/contact_ipm/test_benchmark_protocol.py
                         -v)
        add_test(NAME QuadrupedCITOReplanningSweepParser
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_quadruped_cito_replanning_sweep.py
                         -v)
        set_tests_properties(QuadrupedCITOReplanningSweepParser PROPERTIES
            LABELS quadruped_cito)
    endif()
endif()

option(NMPC_BUILD_CONTACT_BENCHMARKS
       "Build contact-implicit benchmarks reproduced from CRISP" OFF)
if(NMPC_BUILD_CONTACT_BENCHMARKS)
    set(NMPC_CONTACT_PUSH_BOX_NODES 100 CACHE STRING
        "Push Box benchmark node count")
    set(NMPC_CONTACT_PUSH_T_NODES 50 CACHE STRING
        "Push T benchmark node count")
    set(NMPC_CONTACT_PUSH_BOX_DT 0.02 CACHE STRING
        "Push Box benchmark time step")
    set(NMPC_CLOSED_LOOP_PUSH_BOX_NODES 20 CACHE STRING
        "Closed-loop Push Box MPC node count")
    set(NMPC_CLOSED_LOOP_PUSH_BOX_DT 0.1 CACHE STRING
        "Closed-loop Push Box MPC time step")
    set(NMPC_CONTACT_PUSH_T_DT 0.05 CACHE STRING
        "Push T benchmark time step")
    if(NMPC_CONTACT_PUSH_BOX_NODES LESS 2 OR
       NMPC_CLOSED_LOOP_PUSH_BOX_NODES LESS 2 OR
       NMPC_CONTACT_PUSH_T_NODES LESS 2)
        message(FATAL_ERROR "Contact benchmark node counts must be at least 2")
    endif()

    add_executable(contact_cartpole_soft_walls
        benchmarks/contact_ipm/cartpole_soft_walls.cpp)
    target_link_libraries(contact_cartpole_soft_walls PRIVATE nmpc_solver)
    target_compile_definitions(contact_cartpole_soft_walls PRIVATE
        NMPC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(contact_push_box
        benchmarks/contact_ipm/push_box.cpp)
    target_link_libraries(contact_push_box PRIVATE nmpc_solver)
    target_compile_definitions(contact_push_box PRIVATE
        CONTACT_BENCHMARK_NODES=${NMPC_CONTACT_PUSH_BOX_NODES}
        CONTACT_BENCHMARK_DT=${NMPC_CONTACT_PUSH_BOX_DT})
    if(NMPC_BUILD_TESTS)
        add_test(NAME PushBoxContactRegression
                 COMMAND contact_push_box)
    endif()

    add_executable(contact_push_box_closed_loop
        benchmarks/contact_ipm/push_box_closed_loop.cpp)
    target_link_libraries(contact_push_box_closed_loop PRIVATE nmpc_solver)
    target_compile_definitions(contact_push_box_closed_loop PRIVATE
        CONTACT_BENCHMARK_NODES=${NMPC_CLOSED_LOOP_PUSH_BOX_NODES}
        CONTACT_BENCHMARK_DT=${NMPC_CLOSED_LOOP_PUSH_BOX_DT})
    if(NMPC_BUILD_TESTS)
        add_test(NAME PushBoxClosedLoopSmoke
                 COMMAND contact_push_box_closed_loop --mode smoke
                         --output push_box_closed_loop_smoke.json)
    endif()

    add_executable(contact_transport
        benchmarks/contact_ipm/transport.cpp)
    target_link_libraries(contact_transport PRIVATE nmpc_solver)

    add_executable(contact_push_t
        benchmarks/contact_ipm/push_t.cpp)
    target_link_libraries(contact_push_t PRIVATE nmpc_solver)
    target_compile_definitions(contact_push_t PRIVATE
        CONTACT_BENCHMARK_NODES=${NMPC_CONTACT_PUSH_T_NODES}
        CONTACT_BENCHMARK_DT=${NMPC_CONTACT_PUSH_T_DT})
    if(NMPC_BUILD_TESTS)
        add_test(NAME PushTStationarityRegression
                 COMMAND contact_push_t 27)
        add_test(NAME PushTRecoveryRegression
                 COMMAND contact_push_t 25)
        add_test(NAME PushTRawInequalityRegression
                 COMMAND contact_push_t 37)
    endif()
endif()

option(NMPC_BUILD_CASADI_NATIVE_BENCHMARKS
       "Generate and build native CasADi callback benchmarks" OFF)
if(NMPC_BUILD_CASADI_NATIVE_BENCHMARKS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(CASADI_NATIVE_BENCHMARKS
        cart_pendulum
        quadcopter
        quadcopter_tracking
        hanging_chain_2d
        hanging_chain_3d)
    set(CASADI_NATIVE_GENERATED_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/casadi_native_generated")

    foreach(bench ${CASADI_NATIVE_BENCHMARKS})
        set(generated_c
            "${CASADI_NATIVE_GENERATED_DIR}/${bench}_functions.c")
        set(generated_config
            "${CASADI_NATIVE_GENERATED_DIR}/${bench}_config.hpp")
        add_custom_command(
            OUTPUT "${generated_c}" "${generated_config}"
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    "${CASADI_NATIVE_GENERATED_DIR}"
            COMMAND ${Python3_EXECUTABLE} -m casadi_benchmarks.generate_native
                    --problem ${bench}
                    --output-dir "${CASADI_NATIVE_GENERATED_DIR}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks"
            DEPENDS
                benchmarks/casadi_benchmarks/generate_native.py
                benchmarks/casadi_benchmarks/problems.py
                benchmarks/casadi_benchmarks/solvers.py
                benchmarks/casadi_benchmarks/discretize.py
            VERBATIM)
        set_source_files_properties("${generated_c}"
            PROPERTIES GENERATED TRUE LANGUAGE CXX)
        add_executable(casadi_native_${bench}
            benchmarks/casadi_benchmarks/native_benchmark.cpp
            "${generated_c}" "${generated_config}")
        target_include_directories(casadi_native_${bench} PRIVATE
            "${CASADI_NATIVE_GENERATED_DIR}")
        target_compile_definitions(casadi_native_${bench} PRIVATE
            CONTACTIPM_NATIVE_CONFIG_HEADER="${bench}_config.hpp")
        target_link_libraries(casadi_native_${bench} PRIVATE nmpc_solver)
    endforeach()
endif()
option(NMPC_BUILD_FATROP_BENCHMARKS
       "Build optional FATROP-generated benchmarks and diagnostics" OFF)
if(NMPC_BUILD_FATROP_BENCHMARKS)
    # ── Diagnostics ──
    add_executable(diag_verbose benchmarks/fatrop_contactipm/diag_verbose.cpp)
    target_include_directories(diag_verbose PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fatrop_contactipm
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fatrop_codegen/quadcopter_mpc
    )
    target_link_libraries(diag_verbose PRIVATE nmpc_solver)

    add_executable(bench_quadcopter_mpc benchmarks/fatrop_contactipm/bench_quadcopter_mpc.cpp)
    target_include_directories(bench_quadcopter_mpc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fatrop_contactipm
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fatrop_codegen/quadcopter_mpc
    )
    target_link_libraries(bench_quadcopter_mpc PRIVATE nmpc_solver)

    # ── FATROP benchmarks ──
    set(FATROP_BENCHMARKS
        cart_pendulum_mpc
        cart_pendulum_time
        quadcopter_mpc
        quadcopter_p2p
        quadcopter_one_obs
        quadcopter_three_obs
        truck_trailer_time
        hanging_chain_2d
        hanging_chain_3d
    )

    foreach(bench ${FATROP_BENCHMARKS})
        add_executable(fatrop_${bench} benchmarks/fatrop_contactipm/bench_${bench}.cpp)
        target_include_directories(fatrop_${bench} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fatrop_contactipm
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fatrop_codegen/${bench}
        )
        target_link_libraries(fatrop_${bench} PRIVATE nmpc_solver)
    endforeach()

    # Diagnostic executable for cart_pendulum_mpc
    add_executable(diag_cart_pendulum_mpc benchmarks/fatrop_contactipm/diag_cart_pendulum_mpc.cpp)
    target_include_directories(diag_cart_pendulum_mpc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fatrop_contactipm
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fatrop_codegen/cart_pendulum_mpc
    )
    target_link_libraries(diag_cart_pendulum_mpc PRIVATE nmpc_solver)

    # Diagnostic executable for hanging_chain_2d
    add_executable(diag_hanging_chain_2d benchmarks/fatrop_contactipm/diag_hanging_chain_2d.cpp)
    target_include_directories(diag_hanging_chain_2d PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fatrop_contactipm
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fatrop_codegen/hanging_chain_2d
    )
    target_link_libraries(diag_hanging_chain_2d PRIVATE nmpc_solver)
endif()
