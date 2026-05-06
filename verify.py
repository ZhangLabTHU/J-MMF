#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: Qian Lixiang
# @Email: 649811459@qq.com
__author__ = "Qian Lixiang"
__email__ = "649811459@qq.com"

"""
HDkit installation verification script.

Checks that all dependencies are available and all HDkit modules can be
imported successfully. Run this after setting up the conda environment:

    conda activate HDkit
    python verify.py

Exit code 0 = all checks passed; non-zero = failures detected.
"""

import sys
import os
import importlib
import traceback

# ── Colour helpers (ANSI) ──
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

PASS = f"{GREEN}✓{RESET}"
FAIL = f"{RED}✗{RESET}"
WARN = f"{YELLOW}⚠{RESET}"

failures = 0
warnings = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    """Record and print a single check result."""
    global failures
    status = PASS if ok else FAIL
    msg = f"  {status} {label}"
    if detail:
        msg += f"  {detail}"
    print(msg)
    if not ok:
        failures += 1


def warn(msg: str) -> None:
    """Print a warning (does not count as failure)."""
    global warnings
    print(f"  {WARN} {msg}")
    warnings += 1


def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{title}{RESET}")


# ═══════════════════════════════════════════════════════════════════
# 1. Python version
# ═══════════════════════════════════════════════════════════════════
section("1. Python version")

py_ver = sys.version_info
check("Python ≥ 3.10",
      py_ver >= (3, 10),
      f"(detected {py_ver.major}.{py_ver.minor}.{py_ver.micro})")


# ═══════════════════════════════════════════════════════════════════
# 2. Third-party dependencies
# ═══════════════════════════════════════════════════════════════════
section("2. Third-party dependencies")

# NumPy
try:
    import numpy as np
    check("NumPy", True, f"v{np.__version__}")
except ImportError:
    check("NumPy", False, "not found — required by ASE and HDkit")

# ASE
try:
    import ase
    ase_ver = ase.__version__
    ase_ver_tuple = tuple(int(x) for x in ase_ver.split(".")[:2])
    ok_ase = ase_ver_tuple >= (3, 22)
    check("ASE", ok_ase, f"v{ase_ver} (need ≥ 3.22)")
except ImportError:
    check("ASE", False, "not found")
    ase_ver_tuple = (0, 0)

# ASE — NoseHooverChainNVT (the specific NVT integrator used by all examples)
if ase_ver_tuple >= (3, 22):
    try:
        from ase.md.nose_hoover_chain import NoseHooverChainNVT
        check("ASE — NoseHooverChainNVT", True)
    except ImportError:
        check("ASE — NoseHooverChainNVT", False,
              "ASE version looks OK but NoseHooverChainNVT is missing — try updating ASE")
elif ase_ver_tuple > (0, 0):
    check("ASE — NoseHooverChainNVT", False,
          f"requires ASE ≥ 3.22 (detected {ase_ver}) — update with: conda install -c conda-forge 'ase>=3.22'")

# LAMMPS Python bindings
try:
    from lammps import lammps
    check("LAMMPS Python bindings", True)
except ImportError:
    check("LAMMPS Python bindings", False,
          "not found — run `conda install -c conda-forge lammps`")


# ═══════════════════════════════════════════════════════════════════
# 3. HDkit core modules
# ═══════════════════════════════════════════════════════════════════
section("3. HDkit core modules")

# We add the project root (where this script lives) to sys.path so that
# "import HDkit" works without installation.
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

modules_to_check = [
    # (friendly name, import path, class/attr to verify)
    ("HDkit package",            "HDkit",                          None),
    ("HDkit.basin",              "HDkit.basin",                    "BasinManager"),
    ("HDkit.calculators",        "HDkit.calculators",              None),
    ("HDkit.calculators.base",   "HDkit.calculators.basecalculator", "BaseCalculator"),
    ("HDkit.calculators.minmode","HDkit.calculators.minmode",      "MinModeCalculator"),
    ("HDkit.calculators.bondboost","HDkit.calculators.bondboost",  "BondBoostCalculator"),
    ("HDkit.calculators.ridge",  "HDkit.calculators.ridge.mmf",    "MMFPathCalculator"),
]

for name, import_path, attr in modules_to_check:
    try:
        mod = importlib.import_module(import_path)
        if attr is not None:
            cls = getattr(mod, attr, None)
            if cls is None:
                check(name, False, f"module imported but '{attr}' not found")
            else:
                check(name, True)
        else:
            check(name, True)
    except Exception:
        check(name, False, traceback.format_exc().strip().split("\n")[-1])


# ═══════════════════════════════════════════════════════════════════
# 4. Lazy-loading calculator registry
# ═══════════════════════════════════════════════════════════════════
section("4. Lazy-loading calculator registry (HDkit.calculators)")

lazy_checks = [
    ("BondBoostCalculator",),
    ("MinModeCalculator",),
    ("MMFPathCalculator",),
]

for names in lazy_checks:
    primary = names[0]
    ok = False
    err_msg = ""
    for name in names:
        try:
            mod = importlib.import_module("HDkit.calculators")
            obj = getattr(mod, name)
            ok = obj is not None
            if ok:
                break
        except Exception as e:
            err_msg = traceback.format_exc().strip().split("\n")[-1]
    if ok:
        check(f"from HDkit.calculators import {primary}", True)
    else:
        check(f"from HDkit.calculators import {primary}", False, err_msg)


# ═══════════════════════════════════════════════════════════════════
# 5. Example scripts — syntax check
# ═══════════════════════════════════════════════════════════════════
section("5. Example scripts (syntax)")

_run_scripts = ["run_hd.py", "run_compare.py"]

for fname in _run_scripts:
    full = os.path.join(_project_root, fname)
    if not os.path.isfile(full):
        check(fname, False, "file not found")
        continue
    try:
        with open(full, "r", encoding="utf-8") as f:
            source = f.read()
        compile(source, full, "exec")
        check(f"{fname} (syntax)", True)
    except SyntaxError as e:
        check(f"{fname} (syntax)", False, str(e))

# ── Run-script imports ──
# These modules are imported by *every* example script.  Verify them
# separately so users get actionable error messages.
section("5b. Run-script imports (ASE sub-modules)")

run_imports = [
    ("ase.io.read",                       "from ase.io import read"),
    ("ase.io.Trajectory",                 "from ase.io import Trajectory"),
    ("ase.md.nose_hoover_chain.NoseHooverChainNVT",
     "from ase.md.nose_hoover_chain import NoseHooverChainNVT"),
    ("ase.md.velocitydistribution.MaxwellBoltzmannDistribution",
     "from ase.md.velocitydistribution import MaxwellBoltzmannDistribution"),
    ("ase.md.velocitydistribution.Stationary",
     "from ase.md.velocitydistribution import Stationary"),
    ("ase.calculators.lammpslib.LAMMPSlib",
     "from ase.calculators.lammpslib import LAMMPSlib"),
    ("ase.units",                         "from ase import units"),
]

for imp_path, imp_stmt in run_imports:
    try:
        # Use importlib to load the specific symbol
        mod_path, _, attr = imp_path.rpartition(".")
        mod = importlib.import_module(mod_path)
        getattr(mod, attr)
        check(imp_stmt, True)
    except ImportError:
        check(imp_stmt, False,
              "ASE may be too old — update with: conda install -c conda-forge 'ase>=3.22'")
    except AttributeError:
        check(imp_stmt, False,
              "ASE may be too old — update with: conda install -c conda-forge 'ase>=3.22'")


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'─' * 50}{RESET}")
if failures == 0:
    print(f"  {GREEN}{BOLD}All checks passed!{RESET}")
    if warnings:
        print(f"  ({warnings} warning(s) — see above)")
else:
    print(f"  {RED}{BOLD}{failures} check(s) FAILED{RESET}")
    if warnings:
        print(f"  ({warnings} warning(s) — see above)")
print(f"{BOLD}{'─' * 50}{RESET}\n")

sys.exit(0 if failures == 0 else 1)
