#!/usr/bin/env python3
"""Focused regression tests for the independent IMPACT comparison audit."""

from __future__ import annotations

import math
import unittest

import run_impact_audit


class CartAuditTests(unittest.TestCase):
    def test_third_pair_uses_positive_slip(self) -> None:
        states = [[0.0, 0.0, 1.0, 0.0] for _ in range(301)]
        controls = [[0.0, 0.0, 1.0, 0.0] for _ in range(300)]
        metrics = run_impact_audit.audit_cart_transport(
            states, controls, states[0], states[-1]
        )
        self.assertTrue(
            math.isclose(
                metrics["complementarity"], 0.2 * 0.1 * 9.81,
                rel_tol=0.0, abs_tol=1e-14
            )
        )


if __name__ == "__main__":
    unittest.main()
