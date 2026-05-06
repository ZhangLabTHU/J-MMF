# HDkit — Hyperdynamics Example Suite

[English](README.md) | [简体中文](README.zh-cn.md)

A lightweight, installation-free toolkit for **Hyperdynamics (HD) molecular
dynamics** simulations, with a unified runner for Cu(100) surface diffusion.

> **HDkit is a companion code repository for a paper currently under review.**
> It is intended for reproducing and verifying the paper's results —
> **not for production simulations.**

> **Operating System**: This toolkit is designed for **Linux and macOS** only.
> Windows is not supported.  Windows users are recommended to install
> [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows
> Subsystem for Linux) and run the toolkit inside an Ubuntu environment.

This repository accompanies the paper and provides:

- **`HDkit/`** — A streamlined Python package containing the core HD calculators
  (Bond-Boost, MMF, BasinManager).
- **`run_compare.py`** — Compares the bias from all three HD methods on the same
  structure in a single step.
- **`run_hd.py`** — Runs multi-step HD-MD using Bond-Boost, MMF, or J-MMF.

> **Note**: HDkit is a **simplified reference implementation** of the algorithms
> described in our paper.  It focuses on correctness of the core methods; some
> engineering details (error recovery, MPI support, production-grade I/O) are
> deliberately kept minimal.  The simulation lengths in the examples
> (BB: 10 ns; MMF/J-MMF: 100 ps) are chosen so that users can verify the
> code runs end-to-end in minutes to hours — the results reported in
> the paper require substantially longer runs (µs scale) on HPC resources.

---

## Table of Contents

- [Getting Started](#getting-started)
  - [Repository Setup](#repository-setup)
  - [Environment Setup](#environment-setup)
  - [Verify Installation](#verify-installation)
- [Repository Structure](#repository-structure)
- [Bias Comparison (`run_compare.py`)](#bias-comparison-run_comparepy)
- [HD Simulations (`run_hd.py`)](#hd-simulations-run_hdpy)
- [Output Files](#output-files)
- [References](#references)
- [About](#about)
- [License](#license)

---

## Getting Started

### Repository Setup

**Option 1 — Download the latest release** (no Git needed):

> [![Release](https://img.shields.io/github/v/release/ZhangLabTHU/HDkit-example?color=blue)](https://github.com/ZhangLabTHU/HDkit-example/releases)
>
> Go to the [Releases page](https://github.com/ZhangLabTHU/HDkit-example/releases)
> and download `HDkit-example-v1.0.0.zip` or `.tar.gz`.  Extract the
> archive and enter the directory — you are ready to go.

**Option 2 — Clone with Git**:

```bash
git clone https://github.com/ZhangLabTHU/HDkit-example.git
cd HDkit-example
```

> **Results branch**: After running the simulations yourself, you can check out
> the `results` branch to see the pre-computed output files:
>
> ```bash
> git checkout results
> ```

The repository contains everything needed to run the examples:

| File / Directory | Purpose |
|---|---|
| `HDkit/` | Core HD calculator library (no `pip install` needed) |
| `run_hd.py` | Multi-step HD-MD runner |
| `run_compare.py` | Single-step bias comparison |
| `verify.py` | Environment verification |
| `hd-ini.traj` | Initial structure for HD simulations |
| `compare-ini.traj` | Initial structure for bias comparison |
| `Cu_u3.eam` | Cu EAM potential file |
| `CHANGELOG.md` | Release notes |
| `Makefile` | Build release archives (`make release`) |

> **No installation required** — `HDkit/` is a lightweight package that
> lives in the project root.  As long as you run scripts from this
> directory, `import HDkit` works without any path configuration.

### Environment Setup

| Requirement | Minimum Version | Notes |
|---|---|---|
| **Python** | ≥ 3.10 | Uses `match/case` syntax |
| **ASE** | ≥ 3.22 | For `NoseHooverChainNVT` integrator; includes NumPy as a dependency |
| **LAMMPS** | with Python bindings | EAM potential solver |

> **Operating System**: This toolkit is designed for **Linux and macOS**
> only.  Windows users should install
> [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) first.

We recommend a dedicated conda environment.  Create it with one command:

```bash
conda create -n HDkit -c conda-forge python=3.11 ase lammps -y
conda activate HDkit
```

### Verify Installation

```bash
python verify.py
```

If all checks pass, you are ready to run the examples.

---

## Repository Structure

```
.
├── HDkit/                          # ← Lightweight HD toolkit (no install needed)
│   ├── __init__.py                 #   Package init with version info
│   ├── basin.py                    #   BasinManager: basin identification & persistence
│   └── calculators/
│       ├── __init__.py             #   Lazy-loading calculator registry
│       ├── basecalculator.py       #   BaseCalculator: abstract base class
│       ├── minmode.py              #   MinModeCalculator: Hessian minimum-mode (Lanczos)
│       ├── bondboost.py            #   BondBoostCalculator: Bond-Boost HD method
│       └── ridge/
│           ├── __init__.py
│           └── mmf.py              #   MMFPathCalculator: MMF ridge-based HD method
│
├── run_hd.py                       # ← Multi-step HD-MD runner (bb | mmf | j-mmf)
├── run_compare.py                  # ← Single-step bias comparison
├── verify.py                       #   Installation verification script
├── compare-ini.traj                #   Starting structure for bias comparison
├── hd-ini.traj                     #   Starting structure for HD-MD
├── Cu_u3.eam                       #   Cu EAM potential file
├── README.md                       #   English documentation
├── README.zh-CN.md                 #   中文文档
├── CHANGELOG.md                    #   Release notes
└── Makefile                        #   Build release archives (`make release`)
```

After a run, output files are written to a subdirectory (`Climb/`, `Bond-Boost/`, `MMF/`, or `J_MMF/`).

### HDkit Modules

| Module | Class | Purpose |
|---|---|---|
| `basin.py` | `BasinManager` | Identifies and caches energy basins (local minima) via structure optimization; detects saddle points via Hessian eigenvalue analysis |
| `calculators/basecalculator.py` | `BaseCalculator` | Abstract base class providing `std_calc` (unbiased PES) interface and logging |
| `calculators/minmode.py` | `MinModeCalculator` | Computes Hessian minimum eigenvector via Lanczos iteration or full diagonalization; used by MMF to determine climbing direction |
| `calculators/bondboost.py` | `BondBoostCalculator` | Bond-Boost method: monitors bond strains, applies parabolic bias with envelope function that vanishes near transition states |
| `calculators/ridge/mmf.py` | `MMFPathCalculator` | MMF method: climbs along minimum-mode direction to locate energy ridge, applies bias at the saddle-point region. Supports Simple and Shear Jacobian algorithms |

---

## Bias Comparison (`run_compare.py`)

`run_compare.py` applies all three HD methods to the **same starting
structure** (`compare-ini.traj`) and reports the bias energy and force
magnitude produced by each.  This is a single-step evaluation — no MD
integration — designed to let you quickly compare how the three methods
respond to the same atomic configuration.

```bash
python run_compare.py
```

Output is written to `Climb/`:

| File | Content |
|---|---|
| `compare-ini.traj` | Starting structure with std_calc energy & forces |
| `hyper-bb.traj` | Bond-Boost — total (biased) energy & forces |
| `hyper-mmf.traj` | MMF Simple — total energy & forces |
| `hyper-j-mmf.traj` | J-MMF Shear — total energy & forces |
| `bias-bb.traj` | Bond-Boost — bias-only energy & forces |
| `bias-mmf.traj` | MMF Simple — bias-only energy & forces |
| `bias-j-mmf.traj` | J-MMF Shear — bias-only energy & forces |
| `climb-mmf.traj` | MMF Simple — full climbing-path trajectory |
| `climb-j-mmf.traj` | J-MMF Shear — full climbing-path trajectory |

The MMF methods use `emax = −1` (no energy cap) so the climb proceeds
all the way to the ridge.  The hyper trajectories store the full biased
energy and forces; the bias trajectories store only the bias contribution,
making them suitable for visualisation tools that expect standard
energy/force arrays.

**Diagnostic logs** — `run_compare.py` enables verbose output so you can
inspect every step of the calculation:

| Log | Content |
|---|---|
| `rlx.log` | BasinManager — each optimisation step during basin identification |
| `climb.log` | MMF — each climbing step (energy, basin ID, elapsed time) |
| `mode.log` | MinModeCalculator — Lanczos iterations and convergence angles |
| `Bond.log` | BondBoost — basin-update notifications |

In contrast, `run_hd.py` runs with **verbose disabled** to avoid flooding
the disk with millions of log lines during long MD simulations.

---

## HD Simulations (`run_hd.py`)

`run_hd.py` runs full multi-step hyperdynamics MD starting from `hd-ini.traj`.
It demonstrates the complete workflow — equilibration, bias-accelerated
production, and post-processing — for any of the three methods.

> **Before running**: Activate your Python environment and ensure LAMMPS
> is accessible.
>
> ```bash
> conda activate HDkit
> ```

All examples are launched from the **project root**.  No `pip install` or
path configuration is needed — `import HDkit` works because the script
and `HDkit/` live in the same directory.

```bash
python run_hd.py bb        # Bond-Boost
python run_hd.py mmf       # MMF Simple (J_algo="s")
python run_hd.py j-mmf     # J-MMF Shear  (J_algo="h", recommended)
```

The method name is **case-insensitive** — `BB`, `bb`, `Bb` all work.
Output is written to a subdirectory (`Bond-Boost/`, `MMF/`, or `J_MMF/`).

### Method summary

| Argument | Method | emax | Production | loginterval | Key feature |
|---|---|---|---|---|---|
| `bb` | Bond-Boost | 0.3 eV | 10 ns | 10000 | ~1 force-call/step, very efficient |
| `mmf` | MMF Simple | 0.5 eV | 100 ps | 100 | Ridge forces directly (baseline) |
| `j-mmf` | J-MMF Shear | 0.5 eV | 100 ps | 100 | Jacobian propagation + orthogonal projection |

All runs use 500 K, 1 fs timestep, Nose–Hoover chain NVT thermostat,
10 ps unbiased equilibration before switching to the HD calculator.

> **Simulation times**: The default settings are chosen to balance
> turnaround time with statistical quality.  Bond-Boost runs 10 ns
> because it is computationally cheap (~1 force-call per MD step).
> MMF methods run 100 ps because each HD step involves multiple
> force evaluations (climb, Hessian diagonalisation).  All timings
> can be adjusted via `prod_steps` in `run_hd.py`.  The production
> results reported in the paper require substantially longer runs
> (ns–µs scale) on HPC resources.

### Simulation workflow

1. Copy `hd-ini.traj` and `Cu_u3.eam` into the output directory
2. Read structure → set up LAMMPS EAM calculator (unbiased PES)
3. Initialise Maxwell–Boltzmann velocities
4. Equilibrate: NVT at 500 K for 10 ps (unbiased std_calc)
5. Production: NVT HD-MD — 10 ns (BB) or 100 ps (MMF / J-MMF)
6. Post-process: extract basin transitions → compute ACT
7. Convert `hd.traj` → `bias_hd.traj` (bias-only for visualisation)

---

## Output Files

### `run_hd.py` output (in `Bond-Boost/`, `MMF/`, or `J_MMF/`)

Logging is **minimal** to avoid large files during long MD runs.

| File | Description |
|---|---|
| `bias.log` | Per-step bias energy, temperature, and ACT (acceleration factor) |
| `basins.traj` | ASE Trajectory of identified basin (stable state) structures |
| `basins.log` | Summary of basin transitions (frame, distance, moved atoms) |
| `hd.traj` | Full MD trajectory (every frame) |
| `bias_hd.traj` | Bias-only trajectory (bias energy & forces per frame) |
| `HD.log` | ASE MD log (energy, temperature, etc.) |
| `fin.traj` | Final atomic configuration |
| `ini-T.traj` | Structure after equilibration |
| `basin.pkl` | Pickled BasinManager database (for restart) |
| `rlx.log` | BasinManager — final result line per basin identification |
| `Bond.log` | BB internal log (Bond-Boost only) |
| `climb.log` | MMF climbing log (MMF/J_MMF only, final result only) |
| `mode.log` | Lanczos header line (MMF/J_MMF only) |

### `run_compare.py` output (in `Climb/`)

Logging is **verbose** — every optimisation step, climbing iteration,
and Lanczos convergence check is recorded.

| File | Description |
|---|---|
| `hyper-bb.traj` / `hyper-mmf.traj` / `hyper-j-mmf.traj` | Total (biased) energy & forces |
| `bias-bb.traj` / `bias-mmf.traj` / `bias-j-mmf.traj` | Bias-only energy & forces |
| `climb-mmf.traj` / `climb-j-mmf.traj` | Full climbing-path trajectory (MMF only) |
| `compare-ini.traj` | Starting structure with std_calc energy & forces |
| `rlx.log` | BasinManager — every optimisation step per basin identification |
| `climb.log` | MMF — every climbing step (step, energy, basin ID, time) |
| `mode.log` | MinModeCalculator — Lanczos iteration details and convergence |
| `Bond.log` | BondBoost — basin-update notifications |

### Key Metrics

- **ACT** (Accelerated Corrected Time): $\text{ACT} = \exp(\Delta V / k_BT)$, the instantaneous time acceleration factor.
- **HD time**: $\langle\text{ACT}\rangle \times t_\text{wall}$, the effective simulation time accounting for bias.

---

## References

1. Voter, A. F. Hyperdynamics: Accelerated molecular dynamics of infrequent
   events. *Phys. Rev. Lett.* **78**, 3908–3911 (1997).
2. Miron, R. A. & Fichthorn, K. A. Accelerated molecular dynamics with the
   Bond-Boost method. *J. Chem. Phys.* **119**, 6210–6216 (2003).
3. Xiao, P., Duncan, J., Zhang, L. & Henkelman, G. Ridge-based bias potentials
   to accelerate molecular dynamics. *J. Chem. Phys.* **143**, 244104 (2015).

---

## About

### Background

**HDkit** is a lightweight Hyperdynamics toolkit extracted from
**[DLTS](https://github.com/ZhangLabTHU/Hyperdynamics)** (Deep Long Time
Simulation package) — a larger project developed by
[ZhangLab](https://www.zhanglab-thu.com) for long-time-scale dynamics
simulations.  DLTS encompasses multiple methods spanning both molecular
dynamics (Hyperdynamics) and adaptive kinetic Monte Carlo (aKMC).  HDkit
contains only the Hyperdynamics-related code from DLTS, streamlined for
ease of use and reproducibility.

- **ZhangLab homepage**: <https://www.zhanglab-thu.com>
- **ZhangLab on GitHub**: [@ZhangLabTHU](https://github.com/ZhangLabTHU)

### Authorship

The Hyperdynamics code in both HDkit and DLTS was written by
**[PhoenixQian](https://github.com/PhoenixQian)**.

- **Email**: [649811459@qq.com](mailto:649811459@qq.com)
- **GitHub**: <https://github.com/PhoenixQian>

### Paper Status

This repository is a companion to a paper currently under review.
The paper link and recommended citation format will be added here
after publication.

---

## License

This code is provided **solely for verifying and reproducing the results
of the accompanying paper**.  It is a simplified reference implementation
and is **not intended for production molecular dynamics simulations**.
Please cite the relevant references (see above) if you use this toolkit
in your work.
