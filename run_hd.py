#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HDkit — unified hyperdynamics example runner for Cu(100) surface diffusion.

Usage
-----
    python run_hd.py bb          # Bond-Boost
    python run_hd.py mmf         # MMF Simple (J_algo="s")
    python run_hd.py j-mmf       # J-MMF Shear (J_algo="h", recommended)

The method name is case-insensitive: BB, bb, Bb, bB all work.
BB runs 10 ns HD-MD; MMF / J-MMF run 100 ps HD-MD.
All methods run 10 ps unbiased equilibration before hyperdynamics.
Output is written to a subdirectory named after the method.

References
----------
- Voter, A. F. Phys. Rev. Lett. 78, 3908 (1997).
- Miron, R. A. & Fichthorn, K. A. J. Chem. Phys. 119, 6210 (2003).
- Xiao, P., Duncan, J., Zhang, L. & Henkelman, G. J. Chem. Phys. 143, 244104 (2015).
"""

__author__ = "Qian Lixiang"
__email__ = "649811459@qq.com"

import os, sys, shutil
from time import time
import numpy as np

from ase.io import read, write
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units

from HDkit.calculators.bondboost import BondBoostCalculator
from HDkit.calculators.ridge.mmf import MMFPathCalculator


# ═══════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════

def format_line(frame: int, state_id: int, event_id: int,
                moved_atoms: int, distance: float) -> str:
    """Format a single line for basins.log."""
    return (f"{frame:>6d} {state_id:>6d} {event_id:>6d} "
            f"{moved_atoms:>12d} {distance:>10.4f}")


def periodic_distance(pos1: np.ndarray, pos2: np.ndarray,
                      cell: np.ndarray) -> tuple:
    """Minimum-image distance under PBC.

    Returns (total_distance, n_moved_atoms, per_atom_displacements).
    """
    diff = pos2 - pos1
    cell_inv = np.linalg.inv(cell)
    diff_frac = diff @ cell_inv
    diff_frac -= np.round(diff_frac)
    diff = diff_frac @ cell
    atom_dists = np.linalg.norm(diff, axis=1)
    total_dist = np.linalg.norm(diff)
    moved_atoms = int(np.sum(atom_dists > 0.5))
    return total_dist, moved_atoms, atom_dists


def make_lammps_calc() -> LAMMPSlib:
    """Create a LAMMPS EAM calculator for Cu."""
    return LAMMPSlib(
        lammps_header=[
            "units metal",
            "atom_style atomic",
            "atom_modify map array sort 0 0.0",
        ],
        lmpcmds=[
            "pair_style eam",
            "pair_coeff * * Cu_u3.eam",
        ],
        keep_alive=True,
    )


def setup_thermostat(dyn, real_dof: int, temperature_K: float,
                     fixed_indices: list):
    """Fix Nose-Hoover chain DOF for constrained atoms."""
    kT = units.kB * temperature_K
    internal_tdamp = dyn._thermostat._tdamp
    dyn._thermostat._num_atoms_global = real_dof / 3.0
    dyn._thermostat._Q[0] = real_dof * kT * internal_tdamp ** 2

    def kill_fixed_momenta():
        dyn._p[fixed_indices] = 0.0
        atoms = dyn.atoms
        atoms.set_momenta(dyn._p)
    dyn.attach(kill_fixed_momenta)


def post_process(traj_file: str, simulation_time: float,
                 time_unit: str = "ns"):
    """Extract basin transitions and compute ACT from output files."""
    basins_traj = read(traj_file, ":")
    cell = basins_traj[0].get_cell()
    print(f"# Processing {len(basins_traj)} states from {traj_file}")

    with open("basins.log", "w", encoding="utf-8") as summary:
        summary.write("# frame  state  event  moved_atoms   distance\n")
        pos_prev = None
        for event_id, state_atoms in enumerate(basins_traj):
            pos_curr = state_atoms.get_positions()
            if pos_prev is None:
                distance, moved_atoms = 0.0, 0
            else:
                distance, moved_atoms, _ = periodic_distance(
                    pos_prev, pos_curr, cell)
            line = format_line(event_id, event_id, event_id,
                               moved_atoms, distance)
            summary.write(f"{line}\n")
            summary.flush()
            pos_prev = pos_curr

    print(f"# Generated basins.log ({len(basins_traj)} events).")

    # ACT
    try:
        bias_data = np.genfromtxt("bias.log", skip_header=1)
        if bias_data.ndim == 1:
            bias_data = bias_data.reshape(1, -1)
        act_values = bias_data[:, 2]
        ACT_ave = np.mean(act_values)
        hd_time = ACT_ave * simulation_time
        print(f"# ACT from bias.log:")
        print(f"#   Frames : {len(act_values)}")
        print(f"#   ACT_ave: {ACT_ave:.6e}")
        print(f"#   HD time: {hd_time:.6e} {time_unit}")
    except Exception as e:
        print(f"# Failed to compute ACT: {e}")


def convert_to_bias_traj(input_traj: str, output_traj: str) -> None:
    """Convert an ASE trajectory to a bias-only trajectory.

    Reads each frame, extracts ``bias_energy`` and ``bias_forces`` from
    the calculator parameters, and writes them via SinglePointCalculator.
    """
    if not os.path.isfile(input_traj):
        print(f"# Warning: {input_traj} not found, skipping conversion.")
        return
    frames = read(input_traj, ":")
    bias_frames = []
    for atoms in frames:
        bias_e = atoms.calc.parameters.get("bias_energy", 0.0)
        bias_f = atoms.calc.parameters.get(
            "bias_forces", np.zeros_like(atoms.get_positions()))
        atoms_copy = atoms.copy()
        atoms_copy.calc = SinglePointCalculator(
            atoms_copy, energy=bias_e, forces=bias_f)
        bias_frames.append(atoms_copy)
    write(output_traj, bias_frames)
    print(f"# Converted {len(bias_frames)} frames: "
          f"{input_traj} → {output_traj}")


# ═══════════════════════════════════════════════════════════════════════
# method configs
# ═══════════════════════════════════════════════════════════════════════

METHODS = {
    "bb": {
        "dir": "Bond-Boost",
        "label": "Bond-Boost",
        "is_mmf": False,
        "j_algo": None,
        "emax": 0.3,
        "prod_steps": 10000000,       # 10 ns
        "sim_time": 10.0,
        "time_unit": "ns",
        "loginterval": 10000,
        "clean_files": "Bond.log basins.traj bias.log bias_hd.traj mode.log rlx.log",
    },
    "mmf": {
        "dir": "MMF",
        "label": "MMF (Simple)",
        "is_mmf": True,
        "j_algo": "s",
        "emax": 0.5,
        "prod_steps": 100000,         # 100 ps
        "sim_time": 100.0,
        "time_unit": "ps",
        "loginterval": 100,
        "clean_files": "climb.log rlx.log mode.log basins.traj bias_hd.traj",
    },
    "j-mmf": {
        "dir": "J_MMF",
        "label": "J-MMF (Shear)",
        "is_mmf": True,
        "j_algo": "h",
        "emax": 0.5,
        "prod_steps": 100000,         # 100 ps
        "sim_time": 100.0,
        "time_unit": "ps",
        "loginterval": 100,
        "clean_files": "climb.log rlx.log mode.log basins.traj bias_hd.traj",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── parse method ──
    if len(sys.argv) < 2:
        print("Usage: python run_hd.py {bb|mmf|j-mmf}")
        print("  bb     Bond-Boost")
        print("  mmf    MMF Simple (J_algo='s')")
        print("  j-mmf  J-MMF Shear (J_algo='h', recommended)")
        sys.exit(1)

    method_key = sys.argv[1].lower()
    if method_key not in METHODS:
        print(f"Unknown method: '{sys.argv[1]}'")
        print("Valid choices: bb, mmf, j-mmf")
        sys.exit(1)

    cfg = METHODS[method_key]

    # ── setup output directory ──
    script_dir = os.getcwd()
    work_dir = os.path.join(script_dir, cfg["dir"])
    os.makedirs(work_dir, exist_ok=True)

    # copy input files into the work directory
    for fname in ("F19.traj", "Cu_u3.eam"):
        src = os.path.join(script_dir, fname)
        dst = os.path.join(work_dir, fname)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)

    os.chdir(work_dir)

    # ── common simulation parameters ──
    timestep = 1.0 * units.fs
    temperature_K = 500
    tdamp = 100 * units.fs

    # ── read structure ──
    atoms = read("F19.traj")
    real_dof = atoms.get_number_of_degrees_of_freedom()
    fixed_indices = (atoms.constraints[0].get_indices()
                     if atoms.constraints else [])

    calc = make_lammps_calc()

    # ── initial velocities ──
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    Stationary(atoms)

    # ── equilibration (10 ps NVT, unbiased std_calc for all methods) ──
    atoms.calc = calc
    dyn = NoseHooverChainNVT(
        atoms=atoms, timestep=timestep,
        temperature_K=temperature_K, tdamp=tdamp)
    setup_thermostat(dyn, real_dof, temperature_K, fixed_indices)
    dyn.run(steps=10000)            # 10 ps

    atoms.write("ini-T.traj")

    # ── clean stale logs ──
    os.system(f"rm -f {cfg['clean_files']}")

    # ── production ──
    is_mmf = cfg["is_mmf"]
    if is_mmf:
        atoms.calc = MMFPathCalculator(
            std_calc=calc, emax=cfg["emax"], J_algo=cfg["j_algo"],
            temperature_K=temperature_K)
    else:
        atoms.calc = BondBoostCalculator(
            std_calc=calc, emax=cfg["emax"], temperature_K=temperature_K)

    dyn = NoseHooverChainNVT(
        atoms=atoms, timestep=timestep,
        temperature_K=temperature_K, tdamp=tdamp,
        logfile="HD.log", trajectory="hd.traj",
        loginterval=cfg["loginterval"])
    setup_thermostat(dyn, real_dof, temperature_K, fixed_indices)

    print(f"# [{cfg['label']}] Starting production run "
          f"({cfg['prod_steps']} steps)...")
    t0 = time()
    dyn.run(steps=cfg["prod_steps"])
    t1 = time()
    print(f"# [{cfg['label']}] Simulation completed "
          f"in {t1 - t0:.2f} seconds.")

    atoms.write("fin.traj")
    atoms.calc.bm.save_to_file()

    # ── post-process ──
    traj_file = "basins.traj"
    if os.path.isfile(traj_file):
        post_process(traj_file, cfg["sim_time"], cfg["time_unit"])
    else:
        print(f"# Warning: {traj_file} not found, skipping post-process.")

    # ── bias trajectory conversion ──
    convert_to_bias_traj("hd.traj", "bias_hd.traj")