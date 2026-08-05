#!/usr/bin/env python3
"""Canonical entry point for the IMPACT comparison audit."""

from __future__ import annotations

import impact_audit as core


def audit_cart_transport(
    states: list[list[float]],
    controls: list[list[float]],
    start: list[float],
    goal: list[float],
) -> dict[str, float | bool]:
    result = core.audit_cart_transport(states, controls, start, goal)
    friction_limit = 0.2 * 0.1 * 9.81
    complementarity = 0.0
    for control in controls:
        friction_force, _, v, w = control
        g = [v, w, v]
        h = [w, friction_limit - friction_force,
             friction_force + friction_limit]
        complementarity = max(
            complementarity,
            max((abs(left * right) for left, right in zip(g, h)), default=0.0),
        )
    result["complementarity"] = complementarity
    feasible = (
        result["initial_state_error"] <= core.FEASIBILITY_TOL
        and result["dynamics_defect"] <= core.FEASIBILITY_TOL
        and result["side_violation"] <= core.FEASIBILITY_TOL
        and result["equality_violation"] <= core.FEASIBILITY_TOL
        and complementarity <= core.FEASIBILITY_TOL
    )
    result["feasible"] = feasible
    result["audited_success"] = feasible and bool(result["task_success"])
    return result


core.PROBLEMS["cart_transport"] = (4, 4, 300, audit_cart_transport)


if __name__ == "__main__":
    raise SystemExit(core.main())
