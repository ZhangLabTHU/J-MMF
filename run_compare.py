#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HDkit — single-step bias comparison for three HD methods.

Computes the bias energy and forces for the same starting structure using
Bond-Boost, MMF (Simple), and J-MMF (Shear), then writes the results as
ASE trajectories for visual comparison.

Usage
-----
    python run_compare.py

Output (in Climb/)
------------------
compare-ini.traj    Starting structure with std_calc energy & forces
hyper-bb.traj      Bond-Boost    total energy & forces (E_std + E_bias)
hyper-mmf.traj     MMF Simple    total energy & forces
hyper-j-mmf.traj   J-MMF Shear   total energy & forces
bias-bb.traj       Bond-Boost    bias-only energy & forces
bias-mmf.traj      MMF Simple    bias-only energy & forces
bias-j-mmf.traj    J-MMF Shear   bias-only energy & forces
climb-mmf.traj     MMF Simple    climbing-path trajectory
climb-j-mmf.traj   J-MMF Shear   climbing-path trajectory

References
----------
- Voter, A. F. Phys. Rev. Lett. 78, 3908 (1997).
- Miron, R. A. & Fichthorn, K. A. J. Chem. Phys. 119, 6210 (2003).
- Xiao, P., Duncan, J., Zhang, L. & Henkelman, G. J. Chem. Phys. 143, 244104 (2015).
"""

__author__ = "Qian Lixiang"
__email__ = "649811459@qq.com"

import os, shutil
import numpy as np

from ase.io import read, write
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.singlepoint import SinglePointCalculator

from HDkit.calculators.bondboost import BondBoostCalculator
from HDkit.calculators.ridge.mmf import MMFPathCalculator


# ═══════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════

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


def save_hyper_traj(atoms, filename: str) -> None:
    """Save the full biased (hyper) energy and forces as a trajectory."""
    atoms_copy = atoms.copy()
    atoms_copy.calc = SinglePointCalculator(
        atoms_copy,
        energy=atoms.get_potential_energy(),
        forces=atoms.get_forces().copy(),
    )
    write(filename, atoms_copy)


def save_bias_traj(atoms, filename: str) -> None:
    """Save only the bias energy and forces as a trajectory."""
    bias_e = atoms.calc.parameters.get("bias_energy", 0.0)
    bias_f = atoms.calc.parameters.get(
        "bias_forces", np.zeros_like(atoms.get_positions()))
    atoms_copy = atoms.copy()
    atoms_copy.calc = SinglePointCalculator(
        atoms_copy, energy=bias_e, forces=bias_f)
    write(filename, atoms_copy)


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── setup output directory ──
    script_dir = os.getcwd()
    work_dir = os.path.join(script_dir, "Climb")
    os.makedirs(work_dir, exist_ok=True)

    for fname in ("compare-ini.traj", "Cu_u3.eam"):
        src = os.path.join(script_dir, fname)
        dst = os.path.join(work_dir, fname)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)

    os.chdir(work_dir)

    # ── clean stale logs from previous runs ──
    os.system("rm -f Bond.log climb.log climb.traj rlx.log mode.log"
              " compare-ini.traj hyper-*.traj bias-*.traj climb-*.traj")

    # ── step 0: evaluate std_calc on the starting structure ──
    atoms = read("../compare-ini.traj")
    calc = make_lammps_calc()
    atoms.calc = calc
    e_std = atoms.get_potential_energy()
    print(f"# Standard PES energy: {e_std:.6f} eV")
    # save a copy with the computed energy & forces (SinglePointCalculator)
    atoms_std = atoms.copy()
    atoms_std.calc = SinglePointCalculator(
        atoms_std,
        energy=atoms.get_potential_energy(),
        forces=atoms.get_forces().copy())
    write("compare-ini.traj", atoms_std)
    print(f"# Saved std_calc result to compare-ini.traj")

    # ── results table ──
    header = f"  {'Method':<16s} {'E_bias (eV)':>14s} {'|F_bias| (eV/Å)':>18s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # ── 1. Bond-Boost ──
    atoms_bb = atoms.copy()
    atoms_bb.calc = BondBoostCalculator(
        std_calc=calc, emax=0.3, temperature_K=300,
        verbose=True)
    e_bb = atoms_bb.get_potential_energy()
    bias_e_bb = atoms_bb.calc.parameters["bias_energy"]
    bias_f_bb = atoms_bb.calc.parameters["bias_forces"]
    f_bias_bb = np.linalg.norm(bias_f_bb)
    print(f"  {'Bond-Boost':<16s} {bias_e_bb:>14.6f} {f_bias_bb:>18.6f}")
    save_hyper_traj(atoms_bb, "hyper-bb.traj")
    save_bias_traj(atoms_bb, "bias-bb.traj")

    # ── 2. MMF Simple ──
    atoms_ms = atoms.copy()
    atoms_ms.calc = MMFPathCalculator(
        std_calc=calc, emax=-1.0, J_algo="s",
        temperature_K=300, verbose=True,
        mode_log="mode.log",
        write_climb_traj=True)       # emax=-1 → no energy cap
    e_ms = atoms_ms.get_potential_energy()
    bias_e_ms = atoms_ms.calc.parameters["bias_energy"]
    bias_f_ms = atoms_ms.calc.parameters["bias_forces"]
    f_bias_ms = np.linalg.norm(bias_f_ms)
    print(f"  {'MMF Simple':<16s} {bias_e_ms:>14.6f} {f_bias_ms:>18.6f}")
    save_hyper_traj(atoms_ms, "hyper-mmf.traj")
    save_bias_traj(atoms_ms, "bias-mmf.traj")
    # rename the climbing-path trajectory
    if os.path.isfile("climb.traj"):
        os.rename("climb.traj", "climb-mmf.traj")

    # ── 3. J-MMF Shear ──
    atoms_js = atoms.copy()
    atoms_js.calc = MMFPathCalculator(
        std_calc=calc, emax=-1.0, J_algo="h",
        temperature_K=300, verbose=True,
        mode_log="mode.log",
        write_climb_traj=True)       # emax=-1 → no energy cap
    e_js = atoms_js.get_potential_energy()
    bias_e_js = atoms_js.calc.parameters["bias_energy"]
    bias_f_js = atoms_js.calc.parameters["bias_forces"]
    f_bias_js = np.linalg.norm(bias_f_js)
    print(f"  {'J-MMF Shear':<16s} {bias_e_js:>14.6f} {f_bias_js:>18.6f}")
    save_hyper_traj(atoms_js, "hyper-j-mmf.traj")
    save_bias_traj(atoms_js, "bias-j-mmf.traj")
    if os.path.isfile("climb.traj"):
        os.rename("climb.traj", "climb-j-mmf.traj")

    print("  " + "-" * (len(header) - 2))
    print(f"\n# Output written to {work_dir}/")
    print(f"#   compare-ini.traj")
    print(f"#   hyper-bb.traj     hyper-mmf.traj     hyper-j-mmf.traj")
    print(f"#   bias-bb.traj      bias-mmf.traj      bias-j-mmf.traj")
    print(f"#   climb-mmf.traj    climb-j-mmf.traj")
    print(f"#   Bond.log          climb.log          rlx.log          mode.log")