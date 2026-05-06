#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ridge-based hyperdynamics calculators.

Contains MMFPathCalculator which locates the energy ridge (saddle-point region)
by climbing along the Hessian minimum-mode direction, then applies a bias
potential to accelerate rare-event sampling.
"""