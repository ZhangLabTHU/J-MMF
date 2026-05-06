#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Abstract base class for hyperdynamics calculators.

Provides the common interface for all HD calculators: a standard (unbiased)
PES calculator (`std_calc`), force-call counter (`fcalls`), logging, and
the abstract `calculate()` method that derived classes must implement.
"""

__author__ = "Qian Lixiang"
__email__ = "649811459@qq.com"

from contextlib import ExitStack
from typing import Optional, List, Dict, Any
from ase import Atoms
from ase.calculators.calculator import Calculator
from abc import abstractmethod
import os, sys


class DummyComm:
    """Minimal MPI-compatible communicator stub for serial execution."""
    rank = 0


class BaseCalculator(Calculator):
    """
    Abstract base for hyperdynamics calculators.

    Wraps a standard PES calculator (`std_calc`) and provides:

    - `_update_std_results()` — evaluate energy & forces on the unbiased PES.
    - `fcalls` — cumulative count of `std_calc` evaluations.
    - `_log()` / `close()` — per-calculator logging with automatic file handling.

    Subclasses must implement `calculate()`.

    Parameters
    ----------
    std_calc : Calculator
        The unbiased potential-energy-surface calculator (e.g. EMT, LAMMPS).
    logfile : str or file object, optional
        Log destination.  ``"-"`` → stdout, ``None`` → /dev/null.
        Default ``'calc.log'``.
    """

    implemented_properties = ["energy", "forces"]

    std_results: Dict[str, Any] = {"energy": None, "forces": None}

    def __init__(
        self,
        std_calc: Calculator,
        logfile: Optional[str] = 'calc.log',
        comm=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.std_calc = std_calc
        self.fcalls = 0

        self._io = None
        self._exit_stack = ExitStack()
        self._comm = comm or DummyComm()

        if logfile is None:
            logfile = os.devnull

        if logfile == "-":
            self._io = sys.stdout
        elif hasattr(logfile, "write"):
            self._io = logfile
        else:
            if self._comm.rank != 0:
                logfile = os.devnull
            self._io = self._exit_stack.enter_context(
                open(logfile, "a", encoding="utf-8")
            )

    # ── logging ─────────────────────────────────────────────────────

    def close(self):
        """Close the log file (if managed by ExitStack)."""
        if self._exit_stack:
            self._exit_stack.close()

    def _log(self, content: str):
        """Write a line to the log and flush."""
        if hasattr(self._io, "write"):
            self._io.write(f"  {content}\n")
            self._io.flush()

    # ── PES evaluation ──────────────────────────────────────────────

    def _update_std_results(self):
        """Evaluate energy & forces on the unbiased PES (std_calc)."""
        atoms = self.atoms.copy()
        atoms.calc = self.std_calc
        self.std_results = {
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces().copy(),
        }
        self.fcalls += 1

    # ── abstract interface ──────────────────────────────────────────

    @abstractmethod
    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: List[str] = None,
        system_changes: List[str] = None,
    ):
        """Compute biased energy & forces.  Must be overridden."""
        ...
