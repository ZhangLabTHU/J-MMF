#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HDkit calculators sub-package.

Provides lazy-loading access to all HD calculator classes.
Usage:
    from HDkit.calculators import BondBoostCalculator
    from HDkit.calculators import MinModeCalculator
    from HDkit.calculators import MMFPathCalculator
"""

def __getattr__(name):
    """Lazy import to avoid loading all modules at package init time."""

    match name:

        case "MMCalc" | "MinModeCalculator":
            # Hessian minimum-mode calculator (Lanczos / full Hessian)
            from .minmode import MinModeCalculator
            return MinModeCalculator

        case "BBCalc" | "BondBoostCalculator":
            # Bond-Boost hyperdynamics calculator
            from .bondboost import BondBoostCalculator
            return BondBoostCalculator

        case "RDCalc" | "RidgeCalculator" | "MMFPathCalculator":
            # Min-Mode Following (MMF) ridge-based calculator
            from .ridge.mmf import MMFPathCalculator
            return MMFPathCalculator

        case _:
            raise ImportError(f"module {__name__} has no attribute {name}")
