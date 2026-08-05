#!/usr/bin/env python3
"""Add trajectory exports and controlled case parameters to local CRISP."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "// CONTACT_BENCHMARK_INSTRUMENTATION"
CASE_MARKER = "// CONTACT_BENCHMARK_CASE_OVERRIDES"
PUSH_T_SEGMENT_MARKER = "// CONTACT_BENCHMARK_PUSH_T_SEGMENT"
OBJECTIVE_WEIGHTS_MARKER = "// CONTACT_BENCHMARK_OBJECTIVE_WEIGHTS"
SCALING_MARKER = "// CONTACT_BENCHMARK_SCALING"
SCALING_DT_MARKER = "// CONTACT_BENCHMARK_SCALING_DT"

HELPER = r'''
// CONTACT_BENCHMARK_INSTRUMENTATION
std::string contactBenchmarkPath(const std::string& name) {
    const char* directory = std::getenv("CONTACT_BENCHMARK_TRAJECTORY_DIR");
    if (directory == nullptr || directory[0] == '\0') return name;
    return std::string(directory) + "/" + name;
}

void contactBenchmarkSave(const Eigen::VectorXd& x, size_t steps,
                          size_t width, const std::string& name) {
    if (x.size() != static_cast<int>(steps * width))
        throw std::runtime_error("invalid trajectory size");
    std::ofstream output(contactBenchmarkPath(name));
    if (!output) throw std::runtime_error("cannot write trajectory");
    output << std::setprecision(17);
    for (size_t step = 0; step < steps; ++step) {
        for (size_t column = 0; column < width; ++column)
            output << x[step * width + column]
                   << (column + 1 == width ? '\n' : ' ');
    }
}
'''
CASE_HELPER = r'''
// CONTACT_BENCHMARK_CASE_OVERRIDES
void contactBenchmarkOverride(const char* name, Eigen::VectorXd& values) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') return;
    std::stringstream input(raw);
    std::string token;
    for (int index = 0; index < values.size(); ++index) {
        if (!std::getline(input, token, ','))
            throw std::runtime_error(std::string("invalid vector override: ") + name);
        size_t parsed = 0;
        values[index] = std::stod(token, &parsed);
        if (parsed != token.size())
            throw std::runtime_error(std::string("invalid vector override: ") + name);
    }
    if (std::getline(input, token, ','))
        throw std::runtime_error(std::string("invalid vector override: ") + name);
}
'''

OBJECTIVE_WEIGHT_HELPER = r'''
// CONTACT_BENCHMARK_OBJECTIVE_WEIGHTS
scalar_t contactBenchmarkPositiveScale(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') return 1.0;
    char* end = nullptr;
    const scalar_t value = std::strtod(raw, &end);
    if (end == raw || *end != '\0' || !(value > 0.0) ||
        !std::isfinite(value))
        throw std::runtime_error("invalid positive objective scale");
    return value;
}
'''


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one insertion anchor, found {count}")
    return text.replace(old, new)


def instrument_objective_weights(
    text: str, path: Path, output_name: str
) -> str:
    if OBJECTIVE_WEIGHTS_MARKER in text or output_name not in {
        "crisp_push_box.txt",
        "crisp_push_t.txt",
    }:
        return text
    if "#include <cmath>" not in text:
        text = replace_once(
            text, "#include <chrono>\n", "#include <chrono>\n#include <cmath>\n", path
        )
    text = replace_once(
        text,
        "using namespace CRISP;\n",
        "using namespace CRISP;\n" + OBJECTIVE_WEIGHT_HELPER,
        path,
    )
    if output_name == "crisp_push_box.txt":
        for old, new in (
            ("Q(0, 0) = 100;", "Q(0, 0) = 100 * p[3];"),
            ("Q(1, 1) = 100;", "Q(1, 1) = 100 * p[3];"),
            ("Q(2, 2) = 100;", "Q(2, 2) = 100 * p[3];"),
            ("R(0, 0) = 0.001;", "R(0, 0) = 0.001 * p[4];"),
            ("R(1, 1) = 0.001;", "R(1, 1) = 0.001 * p[4];"),
            ("R(2, 2) = 0.001;", "R(2, 2) = 0.001 * p[4];"),
            ("R(3, 3) = 0.001;", "R(3, 3) = 0.001 * p[4];"),
            (
                "auto obj = std::make_shared<ObjectiveFunction>(variableNum, "
                "num_state, problemName, folderName, \"pushboxObjective\", "
                "pushboxObjective);",
                "auto obj = std::make_shared<ObjectiveFunction>(variableNum, "
                "5, problemName, folderName, \"pushboxObjectiveBenchmark\", "
                "pushboxObjective);",
            ),
            (
                "    vector_t xOptimal(variableNum);",
                "    vector_t xOptimal(variableNum);\n"
                "    vector_t objectiveParameters(5);\n"
                "    const scalar_t benchmarkTrackingScale = "
                "contactBenchmarkPositiveScale("
                "\"CONTACT_BENCHMARK_TRACKING_SCALE\");\n"
                "    const scalar_t benchmarkEffortScale = "
                "contactBenchmarkPositiveScale("
                "\"CONTACT_BENCHMARK_EFFORT_SCALE\");",
            ),
            (
                "        solver.setProblemParameters(\"pushboxObjective\", "
                "xFinalStates);",
                "        objectiveParameters << xFinalStates[0], xFinalStates[1],\n"
                "            xFinalStates[2], benchmarkTrackingScale, "
                "benchmarkEffortScale;\n"
                "        solver.setProblemParameters("
                "\"pushboxObjectiveBenchmark\", objectiveParameters);",
            ),
        ):
            text = replace_once(text, old, new, path)
    else:
        for old, new in (
            ("Q(0, 0) = 1;", "Q(0, 0) = p[4];"),
            ("Q(1, 1) = 1;", "Q(1, 1) = p[4];"),
            ("Q(2, 2) = 1;", "Q(2, 2) = p[4];"),
            ("Q(3, 3) = 1;", "Q(3, 3) = p[4];"),
            ("Q_final(0, 0) = 100;", "Q_final(0, 0) = 100 * p[4];"),
            ("Q_final(1, 1) = 100;", "Q_final(1, 1) = 100 * p[4];"),
            ("Q_final(2, 2) = 100;", "Q_final(2, 2) = 100 * p[4];"),
            ("Q_final(3, 3) = 100;", "Q_final(3, 3) = 100 * p[4];"),
            ("R(0, 0) = 0.01;", "R(0, 0) = 0.01 * p[5];"),
            ("R(1, 1) = 0.01;", "R(1, 1) = 0.01 * p[5];"),
            ("R(2, 2) = 0.01;", "R(2, 2) = 0.01 * p[5];"),
            ("R(3, 3) = 0.01;", "R(3, 3) = 0.01 * p[5];"),
            ("R(4, 4) = 0.01;", "R(4, 4) = 0.01 * p[5];"),
            ("R(5, 5) = 0.01;", "R(5, 5) = 0.01 * p[5];"),
            ("R(6, 6) = 0.01;", "R(6, 6) = 0.01 * p[5];"),
            ("R(7, 7) = 0.01;", "R(7, 7) = 0.01 * p[5];"),
            (
                "auto obj = std::make_shared<ObjectiveFunction>(variableNum, 4, "
                "problemName, folderName, \"pushTObjective\", pushTObjective);",
                "auto obj = std::make_shared<ObjectiveFunction>(variableNum, 6, "
                "problemName, folderName, \"pushTObjectiveBenchmark\", "
                "pushTObjective);",
            ),
            (
                "    vector_t xOptimal(variableNum);",
                "    vector_t xOptimal(variableNum);\n"
                "    vector_t objectiveParameters(6);\n"
                "    const scalar_t benchmarkTrackingScale = "
                "contactBenchmarkPositiveScale("
                "\"CONTACT_BENCHMARK_TRACKING_SCALE\");\n"
                "    const scalar_t benchmarkEffortScale = "
                "contactBenchmarkPositiveScale("
                "\"CONTACT_BENCHMARK_EFFORT_SCALE\");",
            ),
            (
                "    solver.setProblemParameters(\"pushTObjective\", "
                "xFinalStates);",
                "    objectiveParameters << xFinalStates[0], xFinalStates[1],\n"
                "        xFinalStates[2], xFinalStates[3], benchmarkTrackingScale,\n"
                "        benchmarkEffortScale;\n"
                "    solver.setProblemParameters(\"pushTObjectiveBenchmark\",\n"
                "                                objectiveParameters);",
            ),
        ):
            text = replace_once(text, old, new, path)
    return text


def instrument_scaling(text: str, path: Path, output_name: str) -> str:
    defaults = {
        "crisp_push_box.txt": (100, "0.02"),
        "crisp_push_t.txt": (50, "0.05"),
    }
    if output_name not in defaults:
        return text
    nodes, time_step = defaults[output_name]
    if SCALING_MARKER not in text:
        text = replace_once(
            text,
            f"const size_t N = {nodes}; // number of time steps",
            f"{SCALING_MARKER}\n"
            "#ifndef CONTACT_BENCHMARK_NODES\n"
            f"#define CONTACT_BENCHMARK_NODES {nodes}\n"
            "#endif\n"
            "static_assert(CONTACT_BENCHMARK_NODES >= 2,\n"
            '              "contact benchmark requires at least two nodes");\n'
            "const size_t N = CONTACT_BENCHMARK_NODES;",
            path,
        )
    if SCALING_DT_MARKER not in text:
        text = replace_once(
            text,
            f"const scalar_t dt = {time_step};",
            f"{SCALING_DT_MARKER}\n"
            "#ifndef CONTACT_BENCHMARK_DT\n"
            f"#define CONTACT_BENCHMARK_DT {time_step}\n"
            "#endif\n"
            "static_assert(CONTACT_BENCHMARK_DT > 0.0,\n"
            '              "contact benchmark requires a positive time step");\n'
            "const scalar_t dt = CONTACT_BENCHMARK_DT;",
            path,
        )
    return text

def instrument_cartpole_guess(text: str, path: Path, output_name: str) -> str:
    if output_name != "crisp_cartpole_soft_walls.txt" or (
        "CONTACT_BENCHMARK_CARTPOLE_GUESS" in text
    ):
        return text
    return replace_once(
        text,
        '    std::string txtFileName = '
        '"/home/workspace/src/examples/pushbot/'
        'initial_guess_pushbot_example.txt";',
        '    const char* benchmarkGuess = '
        'std::getenv("CONTACT_BENCHMARK_CARTPOLE_GUESS");\n'
        '    std::string txtFileName = benchmarkGuess != nullptr\n'
        '        ? benchmarkGuess\n'
        '        : "/home/workspace/src/examples/pushbot/'
        'initial_guess_pushbot_example.txt";',
        path,
    )

def instrument_case_overrides(text: str, path: Path, output_name: str) -> str:
    if CASE_MARKER in text:
        return text
    if "#include <sstream>" not in text:
        text = replace_once(
            text, "#include <iomanip>\n", "#include <iomanip>\n#include <sstream>\n", path
        )
    text = replace_once(
        text, "using namespace CRISP;\n", "using namespace CRISP;\n" + CASE_HELPER, path
    )
    if output_name == "crisp_cartpole_soft_walls.txt":
        text = replace_once(
            text,
            "    xInitialStates << xInitialGuess[0], xInitialGuess[1], "
            "xInitialGuess[2], xInitialGuess[3];",
            "    xInitialStates << xInitialGuess[0], xInitialGuess[1], "
            "xInitialGuess[2], xInitialGuess[3];\n"
            "    contactBenchmarkOverride(\"CONTACT_BENCHMARK_INITIAL_STATE\", "
            "xInitialStates);",
            path,
        )
        text = replace_once(
            text,
            "    xFinalStates << 0,0,0,0;",
            "    xFinalStates << 0,0,0,0;\n"
            "    contactBenchmarkOverride(\"CONTACT_BENCHMARK_TARGET_STATE\", "
            "xFinalStates);",
            path,
        )
    elif output_name == "crisp_push_box.txt":
        text = replace_once(
            text,
            "    xInitialStates << 0, 0, 0;",
            "    xInitialStates << 0, 0, 0;\n"
            "    contactBenchmarkOverride(\"CONTACT_BENCHMARK_INITIAL_STATE\", "
            "xInitialStates);",
            path,
        )
        text = replace_once(
            text,
            "        xFinalStates << 3*cos(theta), 3*sin(theta), theta;",
            "        xFinalStates << 3*cos(theta), 3*sin(theta), theta;\n"
            "        contactBenchmarkOverride(\"CONTACT_BENCHMARK_TARGET_STATE\", "
            "xFinalStates);",
            path,
        )
    elif output_name == "crisp_transport.txt":
        text = replace_once(
            text,
            "    xInitial<< x2_initial + 0.5, x2_initial, -4.0, -4.0, 0.0, 0.0;",
            "    xInitial<< x2_initial + 0.5, x2_initial, -4.0, -4.0, 0.0, 0.0;\n"
            "    contactBenchmarkOverride(\"CONTACT_BENCHMARK_INITIAL_STATE\", "
            "xInitial);",
            path,
        )
        text = replace_once(
            text,
            "    xFinal << x2_final - 0.5, x2_final, -2.0, -2.0, 0.0, 0.0;",
            "    xFinal << x2_final - 0.5, x2_final, -2.0, -2.0, 0.0, 0.0;\n"
            "    contactBenchmarkOverride(\"CONTACT_BENCHMARK_TARGET_STATE\", "
            "xFinal);",
            path,
        )
    return text


def instrument_single(path: Path, output_name: str) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        text = instrument_cartpole_guess(text, path, output_name)
        text = instrument_case_overrides(text, path, output_name)
        text = instrument_objective_weights(text, path, output_name)
        text = instrument_scaling(text, path, output_name)
        path.write_text(text, encoding="utf-8")
        return
    text = replace_once(
        text,
        '#include <chrono>\n',
        '#include <chrono>\n#include <cstdlib>\n#include <fstream>\n'
        '#include <iomanip>\n#include <stdexcept>\n#include <string>\n',
        path,
    )
    text = replace_once(
        text, "using namespace CRISP;\n", "using namespace CRISP;\n" + HELPER, path
    )
    text = replace_once(
        text,
        "    xOptimal = solver.getSolution();",
        "    xOptimal = solver.getSolution();\n"
        f'    contactBenchmarkSave(xOptimal, N, num_state + num_control, '
        f'"{output_name}");',
        path,
    )
    text = instrument_cartpole_guess(text, path, output_name)
    text = instrument_case_overrides(text, path, output_name)
    text = instrument_objective_weights(text, path, output_name)
    text = instrument_scaling(text, path, output_name)
    path.write_text(text, encoding="utf-8")


def instrument_push_t(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER not in text:
        text = replace_once(
            text,
            '#include <chrono>\n',
            '#include <chrono>\n#include <cstdlib>\n',
            path,
        )
        text = replace_once(
            text,
            "using namespace CRISP;\n",
            "using namespace CRISP;\n\n"
            + MARKER
            + "\n"
            + r'''std::string contactBenchmarkPath(const std::string& name) {
    const char* directory = std::getenv("CONTACT_BENCHMARK_TRAJECTORY_DIR");
    if (directory == nullptr || directory[0] == '\0') return name;
    return std::string(directory) + "/" + name;
}
''',
            path,
        )
        text = replace_once(
            text,
            '        segFileName << "pushT_solution_seg_"',
            '        segFileName << "crisp_push_t_seg_"',
            path,
        )
        text = replace_once(
            text,
            "        saveTrajectoryToTextFile(xOptimal, N, num_state + num_control, "
            "segFileName.str());",
            "        saveTrajectoryToTextFile(xOptimal, N, num_state + num_control, "
            "contactBenchmarkPath(segFileName.str()));",
            path,
        )
    if PUSH_T_SEGMENT_MARKER not in text:
        text = replace_once(
            text,
            "    const size_t numSegments = 50;",
            "    const size_t numSegments = 50;\n"
            "    size_t segmentBegin = 0;\n"
            "    size_t segmentEnd = numSegments;\n"
            f"    {PUSH_T_SEGMENT_MARKER}\n"
            "    if (const char* requested = "
            "std::getenv(\"CONTACT_BENCHMARK_PUSH_T_SEGMENT\")) {\n"
            "        char* end = nullptr;\n"
            "        const long value = std::strtol(requested, &end, 10);\n"
            "        if (end == requested || *end != '\\0' || value < 0 ||\n"
            "            value >= static_cast<long>(numSegments))\n"
            "            throw std::runtime_error(\"invalid Push T segment\");\n"
            "        segmentBegin = static_cast<size_t>(value);\n"
            "        segmentEnd = segmentBegin + 1;\n"
            "    }",
            path,
        )
        text = replace_once(
            text,
            "    for (size_t seg = 0; seg < numSegments; ++seg) {",
            "    for (size_t seg = segmentBegin; seg < segmentEnd; ++seg) {",
            path,
        )
    text = instrument_objective_weights(text, path, "crisp_push_t.txt")
    text = instrument_scaling(text, path, "crisp_push_t.txt")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crisp-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "CRISP",
    )
    args = parser.parse_args()
    examples = args.crisp_root.resolve() / "src" / "examples"
    instrument_single(
        examples / "pushbot" / "cpp" / "SolvePushbot.cpp",
        "crisp_cartpole_soft_walls.txt",
    )
    instrument_single(
        examples / "pushbox" / "SolvePushbox.cpp", "crisp_push_box.txt"
    )
    instrument_single(
        examples / "Transp" / "cpp" / "SolveTransp.cpp", "crisp_transport.txt"
    )
    instrument_push_t(examples / "pushT" / "SolvePushT.cpp")
    print(f"CRISP trajectory exports ready under {examples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
