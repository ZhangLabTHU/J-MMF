#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author: Qian Lixiang
# @Email: 649811459@qq.com
__author__ = "Qian Lixiang"
__email__ = "649811459@qq.com"

"""
Min-Mode Following (MMF) Hyperdynamics Calculation Module

This module implements a ridge-based bias-potential calculator for accelerating
rare event sampling. It climbs along the Hessian minimum-mode direction to
locate the potential energy ridge between two basins, then applies a bias
potential to lower the escape barrier.

Core Principles:
1. Identify the current basin
2. Climb along the Hessian minimum-eigenvector direction
3. Locate the potential energy ridge (the dividing surface between two basins)
4. Compute the bias potential and bias forces at the ridge
5. Return corrected energy and forces to the MD simulation

Typical usage:
    >>> from ase.calculators.emt import EMT
    >>> from ase import Atoms
    >>> from ase.md import VelocityVerlet
    >>> 
    >>> # Create an MMF calculator
    >>> mmf_calc = MMFPathCalculator(
    >>>     std_calc=EMT(),
    >>>     emax=0.5,  # Maximum bias energy (eV)
    >>>     stepsize=0.05,  # Climbing step size (Å)
    >>>     temperature_K=300  # Used to compute the boost factor
    >>> )
    >>> 
    >>> atoms = Atoms('Cu4', ...)
    >>> atoms.calc = mmf_calc
    >>> 
    >>> # Run hyperdynamics MD
    >>> dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
    >>> dyn.run(1000)
"""

from typing import List, Optional, Union
from time import time
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import Trajectory
import numpy as np
import gc

from ..minmode import MinModeCalculator
from ...basin import BasinManager

class MMFPathCalculator(MinModeCalculator):
    """
    Min-Mode Following (MMF) Hyperdynamics Bias-Potential Calculator
    
    Climb along the Hessian minimum-mode direction to locate the potential
    energy ridge, then apply a bias potential at the ridge to accelerate
    rare event sampling.  This is one realisation of hyperdynamics, capable
    of boosting the MD time scale by several orders of magnitude.
    
    Algorithm flow:
    1. **Basin identification**: use BasinManager to identify the current basin
    2. **Climbing**: step forward along the minimum-mode direction to seek the
       potential energy ridge
    3. **Ridge determination**:
       - Rollback with step-size reduction detects a local maximum (energy 
         still drops after reducing step) → use the local maximum position as
         the ridge
       - Optimisation converges to a saddle point → current point is the ridge
       - Enter a new basin → midpoint of the last two positions is the ridge
       - Exceeds emax → stop climbing, apply emax as the bias energy
    4. **Bias calculation**:
       - Simple algorithm: directly use the forces at the ridge
       - Shear algorithm: propagate forces back to the initial position via
         the Jacobian chain rule (more accurate)
    5. **Return results**: corrected energy E+ΔE and forces F+ΔF
    
    Parameters
    ----------
    std_calc : Calculator
        Standard PES calculator (e.g., EMT, VASP)
    
    mmcalc : MinModeCalculator, optional
        Custom minimum-mode calculator. If None, a default configuration is
        created automatically.
    
    stepsize : float, default=0.05
        Climbing step size (Å). Distance moved per step along the minimum-mode
        direction.
        - Too small: slow climbing, high computational cost
        - Too large: may overshoot the ridge
        Recommended range: 0.05 ~ 0.2 Å
    
    logfile : str or file object, default='climb.log'
        Climbing process log file. Records energy, basin ID, etc. for each step.
    
    verbose : bool, default=False
        Whether to output detailed climbing logs.
        - False: only output the final result of each climb
        - True: output detailed information for every step
    
    emax : float, default=0.5
        Maximum bias energy (eV). If no ridge is found within emax range,
        stop climbing and apply emax as the bias.
        - Too small: may fail to accelerate certain events
        - Too large: may introduce non-physical behaviour
        Recommended range: 0.5 ~ 2.0 eV (system-dependent)
    
    max_nsteps : int, default=200
        Maximum climbing steps. Prevents infinite climbing.
    
    J_algo : str, default='shear'
        Jacobian propagation algorithm:
        - 'simple' or 's': directly use ridge forces (simple but less accurate)
        - 'shear' or 'h': use the Jacobian chain rule to propagate forces
          (recommended)
        The Shear algorithm accounts for the rotation of the minimum-mode
        direction along the climbing path, yielding more accurate force
        propagation.
    
    temperature_K : float, optional
        System temperature (K). Used to compute the boost factor
        ACT = exp(ΔE/kT).
        If provided, ACT is written to bias.log.
    
    mode_log : str, optional
        Log file path for MinModeCalculator. Defaults to None (no output).
    
    write_basins : bool, default=True
        Whether to write newly discovered basins to 'basins.traj'.
    
    write_climb_traj : bool, default=False
        Whether to write the climbing trajectory to 'climb.traj'.
        For debugging and visualisation.
    
    write_ridges : bool, default=False
        Whether to write ridge structures to 'ridges.traj'.
    
    write_bias_log : bool, default=True
        Whether to write bias energy, temperature, and ACT to 'bias.log'.
    
    bias_interval : int, default=1
        bias.log output interval. Output once every N calculate() calls.
        Used to reduce I/O overhead.
    
    pbc_wrap : bool, default=True
        Whether to wrap atoms into the primary cell after each calculate().
        Recommended to keep on to prevent atoms from "flying out" of the cell.
    
    Attributes
    ----------
    bias_energy : float
        Current bias energy (eV)
    
    bias_forces : np.ndarray
        Current bias forces (eV/Å)
    
    basin_id : int
        ID of the current basin

    ridge_forces : np.ndarray
        Standard forces at the ridge (eV/Å)
    
    Examples
    --------
    Basic usage:
    
    >>> from ase.build import bulk
    >>> from ase.calculators.emt import EMT
    >>> from ase.md import VelocityVerlet
    >>> from ase import units
    >>> 
    >>> # Create atomic structure
    >>> atoms = bulk('Cu', 'fcc', a=3.6).repeat((2, 2, 2))
    >>> 
    >>> # Create an MMF calculator
    >>> mmf_calc = MMFPathCalculator(
    >>>     std_calc=EMT(),
    >>>     emax=0.5,
    >>>     stepsize=0.05,
    >>>     temperature_K=300,
    >>>     verbose=True
    >>> )
    >>> 
    >>> atoms.calc = mmf_calc
    >>> 
    >>> # Run hyperdynamics MD
    >>> dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
    >>> for i in range(1000):
    >>>     dyn.run(10)
    >>>     bias_E = mmf_calc.parameters['bias_energy']
    >>>     print(f"Step {i*10}, Bias energy: {bias_E:.3f} eV")
    
    Adjusting climbing parameters:
    
    >>> mmf_calc = MMFPathCalculator(
    >>>     std_calc=EMT(),
    >>>     emax=0.8,  # Lower maximum bias energy
    >>>     stepsize=0.1,  # Larger step size for faster climbing
    >>>     max_nsteps=200,  # Increase maximum steps
    >>>     J_algo='shear',  # Use more accurate Jacobian propagation
    >>> )
    
    Using a custom MinModeCalculator:
    
    >>> mmcalc = MinModeCalculator(
    >>>     std_calc=EMT(),
    >>>     algo='Vibration',  # Use the Vibration method
    >>>     tolerance=1e-3,
    >>> )
    >>> mmf_calc = MMFPathCalculator(
    >>>     std_calc=EMT(),
    >>>     mmcalc=mmcalc,
    >>> )
    
    Notes
    -----
    - The energy and forces returned by calculate() already include the bias
      terms: E_total = E0 + ΔE, F_total = F0 + ΔF
    - The bias energy is usually positive, lowering the escape barrier
    - ACT (Accelerated Corrected Time) represents the time boost factor
    - If the system is near a saddle point (basin_id=None), no bias is applied
    - The climbing process automatically handles periodic boundary conditions
    - BasinManager is used for basin identification; its output goes to
      'rlx.log'
    - The Shear algorithm costs the same as Simple but is more accurate;
      it is recommended
    
    References
    ----------
    .. [1] Voter, A. F. (1997). Hyperdynamics: Accelerated molecular dynamics 
           of infrequent events. Physical Review Letters, 78(20), 3908.
    .. [2] Miron, R. A., & Fichthorn, K. A. (2003). Accelerated molecular 
           dynamics with the bond-boost method. The Journal of Chemical 
           Physics, 119(12), 6210-6216.
    """

    default_parameters = {
        # Results storage
        **MinModeCalculator.default_parameters,
        "basin_id": -1,
        "bias_energy": None,
        "bias_forces": None,
        "ridge_forces": None,
        "fcalls": 0,
    }

    def __init__(
        self,
        std_calc: Calculator,
        # Calculator settings
        J_algo: str = "shear",
        emax: float = 0.5,
        max_nsteps: int = 200,
        temperature_K: Optional[float] = None,
        stepsize: float = 0.05,
        # Output settings
        mode_log: Optional[Union[str, object]] = None,
        write_basins: bool = True,
        write_climb_traj: bool = False,
        write_ridges: bool = False,
        write_bias_log: bool = True,
        # Other settings
        logfile: Optional[Union[str, object]] = "climb.log",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            std_calc = std_calc,
            logfile = logfile,
            **kwargs,
        )

        match J_algo.lower():
            case "simple" | "s":
                self._J_algo = "s"
            case "shear" | "h":
                self._J_algo = "h"
            case _:
                raise ValueError(f"Unknown J_algo: {J_algo}")
        self.emax = emax
        self.max_nsteps = max_nsteps
        self._temperature_K = temperature_K
        self._stepsize = stepsize
        self.mm_calc = MinModeCalculator(
            std_calc = self.std_calc,
            algo = self._algo,
            n_eigs = self._n_eigs,
            delta = self._delta,
            tolerance = self._tolerance,
            direction = self._direction,
            orth = self._orth,
            max_niter = self._max_niter,
            logfile = mode_log if mode_log is not None else None,
        )

        self.bm = BasinManager(verbose=verbose)
        self._verbose = verbose

        class DummyTrajectoryWriter:
            def write(self, *args, **kwargs):
                pass

        class DummyLogWriter:
            def write(self, *args, **kwargs):
                pass
            def flush(self):
                pass
            def tell(self):
                return -1

        self.write_basins = (
            self._exit_stack.enter_context(Trajectory("basins.traj", "a"))
            if write_basins
            else DummyTrajectoryWriter()
        )
        self.write_climb_traj = (
            self._exit_stack.enter_context(Trajectory("climb.traj", "a"))
            if write_climb_traj
            else DummyTrajectoryWriter()
        )
        self.write_ridges = (
            self._exit_stack.enter_context(Trajectory("ridges.traj", "a"))
            if write_ridges
            else DummyTrajectoryWriter()
        )
        self.bias_log = (
            self._exit_stack.enter_context(open("bias.log", "a", encoding="utf-8"))
            if write_bias_log
            else DummyLogWriter()
        )

    def _climb_log(
        self,
        step: int,
        energy: float,
        basin_id: Optional[int] = None,
        init: bool = False,
        is_final: bool = False,
    ):
        """Log one climbing step.

        When ``init=True``, writes the header line.  When ``is_final=True``,
        the line is always written; otherwise it is written only if
        ``verbose=True``.
        """
        if init:
            self.start_time = time()
            self._log(f"# {'Step':>4} {'Time':>8} {'Energy':>15} {'Basin ID':>12}")
            return

        if not self._verbose and not is_final:
            return

        elapsed = time() - self.start_time
        elapsed_seconds = int(round(elapsed))
        hours = elapsed_seconds // 3600
        minutes = (elapsed_seconds % 3600) // 60
        seconds = elapsed_seconds % 60
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        if basin_id is None:
            basin_str = " " * 12
        else:
            basin_str = f"{basin_id:>12d}"
        self._log(f"{step:>6d} {time_str} {energy:>15.6f} {basin_str}")

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: List[str] = None,
        system_changes: List[str] = all_changes,
    ):
        """
        Compute the MMF bias energy and forces.

        Workflow
        --------
        1. Identify the current basin via BasinManager.
        2. Climb along the Hessian minimum-mode direction to locate the ridge.
        3. Compute the bias: ΔE = E_ridge − E_basin, ΔF propagated via Jacobian.
        4. Return total energy = E_basin + ΔE and forces = F_basin + ΔF.

        Stop conditions (priority order):
          1. Local maximum detected (rollback with step-size reduction).
          2. Bias exceeds *emax*.
          3. Optimization converges to a saddle point.
          4. A new basin is entered (ridge = midpoint of last two positions).
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        if properties is None:
            properties = self.implemented_properties

        # ── step 1: evaluate unbiased energy & forces ──
        self._update_std_results()
        e0 = self.std_results["energy"]
        F0 = self.std_results["forces"]

        # work on a copy to avoid side effects
        atoms = self.atoms.copy()
        atoms.calc = self.std_calc
        pos0 = atoms.get_positions()

        # ── step 2: identify current basin ──
        basin_id_0 = self.bm.map_atoms_to_basin(atoms)

        if basin_id_0 is None:
            # near a saddle point — do not apply bias
            self._climb_log(step=0, energy=0.0, basin_id=-1, init=True, is_final=True)
            self._log("# At saddle point, no bias applied")
            self.results["energy"] = e0
            self.results["forces"] = F0
            self.parameters["bias_energy"] = 0.0
            self.parameters["bias_forces"] = np.zeros_like(F0)
            self.fcalls += self.bm.fcalls
            self.bm.fcalls = 0
            self.parameters["fcalls"] = self.fcalls
            return

        # write new basin structure if it changed
        if basin_id_0 != self.parameters["basin_id"]:
            self.write_basins.write(self.bm.export_basin(basin_id_0))
        self.parameters["basin_id"] = basin_id_0

        # ── step 3: determine initial climbing direction N_m ──
        # climb_atoms carries the min-mode calculator; get_forces() returns
        # the lowest-eigenvector direction, normalised.
        climb_atoms = atoms.copy()
        mm_calc = self.mm_calc
        mm_calc.parameters["min_mode"] = self.parameters["min_mode"]
        climb_atoms.calc = mm_calc
        N_m = climb_atoms.get_forces()
        self.parameters["min_mode"] = mm_calc.parameters["min_mode"]

        # ── step 4: climb toward the ridge ──
        pos = pos0.copy()
        current_basin_id = basin_id_0
        emax = np.inf if self.emax < 0 else self.emax
        climb_atoms.calc.results["forces"] = N_m
        self.write_climb_traj.write(climb_atoms)
        self._climb_log(step=0, energy=0.0, basin_id=basin_id_0, init=True)
        N_m_list = []
        ridge_atoms = None
        e_old = e0          # accepted energy from the previous step

        for m in range(1, self.max_nsteps + 1):

            if self._J_algo == "h":
                N_m_list.append(N_m)

            # ── take a step along the current min-mode direction ──
            pos_old = pos.copy()
            pos = pos_old + N_m * self._stepsize
            atoms.set_positions(pos)
            e = atoms.get_potential_energy()
            self.fcalls += 1

            # ── rollback & step-size reduction: detect energy drop ──
            if e < e_old:
                # retry with 1/10 step
                stepsize_tmp = self._stepsize / 10.0
                pos_temp = pos_old + N_m * stepsize_tmp
                atoms.set_positions(pos_temp)
                e_temp = atoms.get_potential_energy()
                self.fcalls += 1

                if e_temp < e_old:
                    # even the reduced step lost energy → crossed the local
                    # maximum; pos_old is the ridge
                    self._climb_log(
                        step=m,
                        energy=e_old - e0,
                        basin_id=basin_id_0,
                        is_final=True,
                    )
                    self._log("# Ridge found (local maximum, rollback to pos_old)")
                    ridge_atoms = atoms.copy()
                    ridge_atoms.set_positions(pos_old)
                    break
                else:
                    # reduced step is accepted
                    pos = pos_temp
                    e = e_temp
                    atoms.set_positions(pos)

            e_old = e  # step accepted

            # ── stop condition 1: bias exceeds emax ──
            if e - e0 > emax:
                self._climb_log(
                    step=m,
                    energy=e - e0,
                    basin_id=None,
                    is_final=True,
                )
                self._log("# Reached maximum bias energy")
                break

            # ── stop condition 2: converged to a saddle point ──
            current_basin_id = self.bm.map_atoms_to_basin(atoms)
            if current_basin_id is None:
                self._climb_log(
                    step=m,
                    energy=e - e0,
                    basin_id=-1,
                    is_final=True,
                )
                self._log("# Ridge found (saddle point)")
                current_basin_id = basin_id_0
                ridge_atoms = atoms.copy()
                break

            # ── stop condition 3: entered a new basin ──
            if current_basin_id != basin_id_0:
                self._climb_log(
                    step=m,
                    energy=e - e0,
                    basin_id=current_basin_id,
                    is_final=True,
                )
                self._log("# Ridge found (new basin)")
                # ridge = midpoint between last two positions
                pos = (atoms.get_positions() + pos_old) / 2
                ridge_atoms = atoms.copy()
                ridge_atoms.set_positions(pos)
                break

            # ── still climbing: compute new N_m ──
            if m == self.max_nsteps:
                self._climb_log(
                    step=m,
                    energy=e - e0,
                    basin_id=current_basin_id,
                    is_final=True,
                )
                self._log("# Reached maximum climb steps")
            else:
                self._climb_log(
                    step=m,
                    energy=e - e0,
                    basin_id=current_basin_id,
                )

            pos_old = pos.copy()
            climb_atoms.set_positions(pos)
            N_m_old = N_m.copy()
            N_m = climb_atoms.get_forces()
            # flip sign if the new mode points opposite to the previous one
            if np.vdot(N_m, N_m_old) < 0:
                N_m = -N_m

            climb_atoms.calc.results["forces"] = N_m
            self.write_climb_traj.write(climb_atoms)

        # ── step 5: compute bias energy & forces ──
        if ridge_atoms is None:
            # ridge not found — apply maximum bias
            self._log("# Ridge not found, applying emax")
            e_total = e0 + emax
            F_total = F0
        else:
            self._log("# Ridge found, calculating bias")
            ridge_atoms.calc = self.std_calc
            e_total = ridge_atoms.get_potential_energy()
            self.fcalls += 1
            if e_total < e0:
                self._log("WARNING: Ridge energy lower than initial energy")
                self._log("WARNING: Apply zero bias instead")
                e_total = e0
                F_total = F0
            else:
                F_ridge = ridge_atoms.get_forces()
                self.parameters["ridge_forces"] = F_ridge.copy()
                self.write_ridges.write(ridge_atoms)

                if self._J_algo == "s":
                    # Simple algorithm (naive baseline): use ridge forces directly,
                    # without orthogonal projection.  The non-physical component
                    # along the minimum-mode direction is kept for comparison.
                    F_total = F_ridge

                elif self._J_algo == "h":
                    # Shear algorithm: propagate ridge forces back to the initial
                    # position via the chain of Jacobian matrices, then remove
                    # the non-physical component along N̂_0 by orthogonal projection.
                    ridge_atoms.calc = mm_calc
                    N_m_old = N_m.copy()
                    N_m = ridge_atoms.get_forces()
                    if np.vdot(N_m, N_m_old) < 0:
                        N_m = -N_m
                    N_m_list.append(N_m)

                    # Jacobian transpose: F_hyper_raw = J_total^T @ F_ridge
                    #   J_m^T @ x = x + N_m · ⟨N_{m+1} − N_m, x⟩
                    F_curr = F_ridge.reshape(-1)

                    for m in range(len(N_m_list) - 2, -1, -1):
                        nm = N_m_list[m].reshape(-1)
                        nm_plus_1 = N_m_list[m + 1].reshape(-1)
                        u = nm_plus_1 - nm
                        v = nm
                        dot_val = np.dot(u, F_curr)
                        F_curr = F_curr + v * dot_val

                    F_hyper_raw = F_curr.reshape(-1, 3)

                    # Orthogonal projection (Shear only): remove the component
                    # of F_hyper_raw along N̂_0 to avoid biasing the reaction
                    # coordinate directly (statistical correctness).
                    #   F_hyper_final = F_hyper_raw − (F_hyper_raw · N̂_0) N̂_0
                    #   F_bias        = F_hyper_final − F_std
                    N_0_flat = self.parameters["min_mode"].reshape(-1)
                    N_0_norm = np.linalg.norm(N_0_flat)
                    if N_0_norm > 1e-10:
                        N_0_unit = N_0_flat / N_0_norm
                        F_hyper_flat = F_hyper_raw.reshape(-1)
                        c = np.dot(F_hyper_flat, N_0_unit)
                        F_hyper_final = F_hyper_flat - c * N_0_unit
                        F_total = F_hyper_final.reshape(-1, 3)
                    else:
                        F_total = F_hyper_raw

        # ── step 6: store results ──
        self.results["energy"] = e_total
        self.results["forces"] = F_total
        self.parameters["bias_energy"] = e_total - e0
        self.parameters["bias_forces"] = F_total - F0
        self.fcalls += self.mm_calc.fcalls
        self.mm_calc.fcalls = 0
        self.fcalls += self.bm.fcalls
        self.bm.fcalls = 0
        self.parameters["fcalls"] = self.fcalls

        # ── step 7: write bias.log ──
        if self.bias_log.tell() == 0:
            self.bias_log.write(
                "# bias_energy    temperature    ACT\n"
            )

        bias_energy = e_total - e0
        if self._temperature_K is not None and self._temperature_K > 0:
            act = np.exp(bias_energy / (units.kB * self._temperature_K))
            self.bias_log.write(
                f"{bias_energy:>12.6f} {self._temperature_K:>12.1f} {act:>15.6e}\n"
            )
        else:
            self.bias_log.write(
                f"{bias_energy:>12.6f} {' ':>12s} {' ':>15s}\n"
            )
        self.bias_log.flush()

        # Clean up memory
        gc.collect()