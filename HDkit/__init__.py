#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HDkit — Lightweight Hyperdynamics Toolkit

A streamlined, installation-free Python package for Hyperdynamics (HD) molecular
dynamics simulations. Extracted from the DLTS project, retaining only the modules
essential for HD workflows.

Core Components
---------------
- BondBoostCalculator : Bond-Boost hyperdynamics (Miron & Fichthorn, 2003)
- MMFPathCalculator   : Min-Mode Following ridge-based HD (Xiao et al., 2015)
- BasinManager        : Energy basin identification and persistence

Quick Start
-----------
    from HDkit.calculators.bondboost import BondBoostCalculator
    from HDkit.calculators.ridge.mmf import MMFPathCalculator
    from HDkit.basin import BasinManager
"""

__version__ = "1.0.0"
