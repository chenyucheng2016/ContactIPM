#ifndef CONTACT_IPM_TRAJECTORY_OUTPUT_HPP
#define CONTACT_IPM_TRAJECTORY_OUTPUT_HPP

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace contact_benchmark {

inline std::ofstream trajectory_file(const char* solver, const char* problem,
                                     int segment = -1) {
    const char* directory =
        std::getenv("CONTACT_BENCHMARK_TRAJECTORY_DIR");
    if (!directory || directory[0] == '\0') return {};

    std::ostringstream path;
    path << directory << '/' << solver << '_' << problem;
    if (segment >= 0)
        path << "_seg_" << std::setfill('0') << std::setw(2) << segment;
    path << ".txt";
    std::ofstream output(path.str());
    output << std::setprecision(17);
    return output;
}

} // namespace contact_benchmark

#endif
