#pragma once

#include "examples/quadruped_cito/quadruped_cito_model.hpp"

namespace quadruped_cito {

enum class TerrainKind {
    FLAT,
    SLOPE,
    SINUSOIDAL,
    SMOOTH_STEP,
    RANDOM_SMOOTH
};

// A single analytic terrain description used both by CITO and by the MuJoCo
// height-field generator.  MuJoCo samples this function on its collision grid;
// CITO evaluates the same function and its analytic derivatives.
struct SharedTerrain {
    static constexpr bool kAffineFrame = false;

    TerrainKind kind = TerrainKind::FLAT;
    double offset = 0.0;
    double slope_x = 0.0;
    double slope_y = 0.0;
    double amplitude = 0.0;
    double wave_number_x = 0.0;
    double wave_number_y = 0.0;
    double phase_x = 0.0;
    double phase_y = 0.0;
    double secondary_amplitude = 0.0;
    double secondary_wave_number_x = 0.0;
    double secondary_wave_number_y = 0.0;
    double secondary_phase_x = 0.0;
    double secondary_phase_y = 0.0;
    double step_height = 0.0;
    double step_center_x = 0.0;
    double step_sharpness = 0.0;

    static SharedTerrain flat() { return SharedTerrain{}; }

    static SharedTerrain slope(double x_slope, double y_slope = 0.0) {
        SharedTerrain terrain;
        terrain.kind = TerrainKind::SLOPE;
        terrain.slope_x = x_slope;
        terrain.slope_y = y_slope;
        return terrain;
    }

    static SharedTerrain sinusoidal(double wave_amplitude = 0.02,
                                    double x_wave_number = 5.0,
                                    double y_wave_number = 3.0) {
        SharedTerrain terrain;
        terrain.kind = TerrainKind::SINUSOIDAL;
        terrain.amplitude = wave_amplitude;
        terrain.wave_number_x = x_wave_number;
        terrain.wave_number_y = y_wave_number;
        return terrain;
    }

    static SharedTerrain smooth_step(double height = 0.03,
                                     double center_x = 0.24,
                                     double sharpness = 18.0) {
        SharedTerrain terrain;
        terrain.kind = TerrainKind::SMOOTH_STEP;
        terrain.step_height = height;
        terrain.step_center_x = center_x;
        terrain.step_sharpness = sharpness;
        return terrain;
    }

    static SharedTerrain random_smooth(unsigned int seed,
                                       double maximum_amplitude = 0.02) {
        unsigned int state = seed ^ 0x9e3779b9U;
        const auto unit = [&state]() {
            state = 1664525U * state + 1013904223U;
            return static_cast<double>((state >> 8) & 0x00ffffffU) /
                   static_cast<double>(0x01000000U);
        };
        constexpr double two_pi = 6.28318530717958647692;
        SharedTerrain terrain;
        terrain.kind = TerrainKind::RANDOM_SMOOTH;
        terrain.amplitude = 0.65 * maximum_amplitude;
        terrain.wave_number_x = 4.0 + 2.0 * unit();
        terrain.wave_number_y = 2.5 + 2.0 * unit();
        terrain.phase_x = two_pi * unit();
        terrain.phase_y = two_pi * unit();
        terrain.secondary_amplitude = 0.35 * maximum_amplitude;
        terrain.secondary_wave_number_x = 7.0 + 3.0 * unit();
        terrain.secondary_wave_number_y = 4.0 + 2.0 * unit();
        terrain.secondary_phase_x = two_pi * unit();
        terrain.secondary_phase_y = two_pi * unit();
        return terrain;
    }

    double height(double x, double y) const {
        const double plane = offset + slope_x * x + slope_y * y;
        if (kind == TerrainKind::SINUSOIDAL ||
            kind == TerrainKind::RANDOM_SMOOTH) {
            const double primary = amplitude *
                std::sin(wave_number_x * x + phase_x) *
                std::sin(wave_number_y * y + phase_y);
            const double secondary = secondary_amplitude *
                std::sin(secondary_wave_number_x * x + secondary_phase_x) *
                std::sin(secondary_wave_number_y * y + secondary_phase_y);
            return plane + primary + secondary;
        }
        if (kind == TerrainKind::SMOOTH_STEP) {
            return plane + 0.5 * step_height *
                               (1.0 + std::tanh(step_sharpness *
                                                (x - step_center_x)));
        }
        return plane;
    }

    bool sample(const Vec<3>& position, TerrainSample& result) const {
        const double x = position[0];
        const double y = position[1];
        double terrain_height = offset + slope_x * x + slope_y * y;
        double hx = slope_x;
        double hy = slope_y;
        double hxx = 0.0;
        double hxy = 0.0;
        double hyy = 0.0;
        if (kind == TerrainKind::SINUSOIDAL ||
            kind == TerrainKind::RANDOM_SMOOTH) {
            const double sx = std::sin(wave_number_x * x + phase_x);
            const double cx = std::cos(wave_number_x * x + phase_x);
            const double sy = std::sin(wave_number_y * y + phase_y);
            const double cy = std::cos(wave_number_y * y + phase_y);
            terrain_height += amplitude * sx * sy;
            hx += amplitude * wave_number_x * cx * sy;
            hy += amplitude * wave_number_y * sx * cy;
            hxx -= amplitude * wave_number_x * wave_number_x * sx * sy;
            hxy += amplitude * wave_number_x * wave_number_y * cx * cy;
            hyy -= amplitude * wave_number_y * wave_number_y * sx * sy;
            const double sx2 = std::sin(
                secondary_wave_number_x * x + secondary_phase_x);
            const double cx2 = std::cos(
                secondary_wave_number_x * x + secondary_phase_x);
            const double sy2 = std::sin(
                secondary_wave_number_y * y + secondary_phase_y);
            const double cy2 = std::cos(
                secondary_wave_number_y * y + secondary_phase_y);
            terrain_height += secondary_amplitude * sx2 * sy2;
            hx += secondary_amplitude * secondary_wave_number_x * cx2 * sy2;
            hy += secondary_amplitude * secondary_wave_number_y * sx2 * cy2;
            hxx -= secondary_amplitude * secondary_wave_number_x *
                   secondary_wave_number_x * sx2 * sy2;
            hxy += secondary_amplitude * secondary_wave_number_x *
                   secondary_wave_number_y * cx2 * cy2;
            hyy -= secondary_amplitude * secondary_wave_number_y *
                   secondary_wave_number_y * sx2 * sy2;
        } else if (kind == TerrainKind::SMOOTH_STEP) {
            const double transition =
                std::tanh(step_sharpness * (x - step_center_x));
            const double sech_sq = 1.0 - transition * transition;
            terrain_height += 0.5 * step_height * (1.0 + transition);
            hx += 0.5 * step_height * step_sharpness * sech_sq;
            hxx = -step_height * step_sharpness * step_sharpness *
                  sech_sq * transition;
        }
        return sample_height_field(position, terrain_height, hx, hy, hxx,
                                   hxy, hyy, result);
    }
};

inline const char* terrain_name(TerrainKind kind) {
    switch (kind) {
        case TerrainKind::FLAT: return "flat";
        case TerrainKind::SLOPE: return "slope";
        case TerrainKind::SINUSOIDAL: return "sinusoidal";
        case TerrainKind::SMOOTH_STEP: return "smooth_step";
        case TerrainKind::RANDOM_SMOOTH: return "random_smooth";
    }
    return "unknown";
}

}  // namespace quadruped_cito
