#pragma once

#include <cctype>
#include <cstdio>
#include <cstdlib>

#include "nmpc/nmpc_core.hpp"

namespace contact_benchmark {

[[noreturn]] inline void invalid_vector_override(const char* name) {
    std::fprintf(stderr, "invalid vector override: %s\n", name);
    std::exit(2);
}

template <int N>
bool apply_vector_override(const char* name, nmpc::Vec<N>& values) {
    const char* cursor = std::getenv(name);
    if (cursor == nullptr || cursor[0] == '\0') return false;

    for (int index = 0; index < N; ++index) {
        while (std::isspace(static_cast<unsigned char>(*cursor))) ++cursor;
        char* end = nullptr;
        const double value = std::strtod(cursor, &end);
        if (end == cursor)
            invalid_vector_override(name);
        values[index] = value;
        cursor = end;
        while (std::isspace(static_cast<unsigned char>(*cursor))) ++cursor;
        if (index + 1 < N) {
            if (*cursor != ',')
                invalid_vector_override(name);
            ++cursor;
        }
    }
    while (std::isspace(static_cast<unsigned char>(*cursor))) ++cursor;
    if (*cursor != '\0')
        invalid_vector_override(name);
    return true;
}

template <int N>
bool apply_vector_prefix_override(const char* name, nmpc::Vec<N>& values,
                                  int supplied_size) {
    const char* cursor = std::getenv(name);
    if (cursor == nullptr || cursor[0] == '\0') return false;
    if (supplied_size < N)
        invalid_vector_override(name);

    for (int index = 0; index < supplied_size; ++index) {
        while (std::isspace(static_cast<unsigned char>(*cursor))) ++cursor;
        char* end = nullptr;
        const double value = std::strtod(cursor, &end);
        if (end == cursor)
            invalid_vector_override(name);
        if (index < N)
            values[index] = value;
        cursor = end;
        while (std::isspace(static_cast<unsigned char>(*cursor))) ++cursor;
        if (index + 1 < supplied_size) {
            if (*cursor != ',')
                invalid_vector_override(name);
            ++cursor;
        }
    }
    while (std::isspace(static_cast<unsigned char>(*cursor))) ++cursor;
    if (*cursor != '\0')
        invalid_vector_override(name);
    return true;
}

} // namespace contact_benchmark
