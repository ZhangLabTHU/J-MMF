# Changelog

All notable changes to the HDkit-example project are documented in this file.

---

## [v1.0.0] — 2026-05-06

### First Public Release 🎉

This is the initial release of **HDkit-example**, a lightweight, installation-free
toolkit for Hyperdynamics (HD) molecular dynamics simulations, accompanying the
paper on Cu(100) surface diffusion.

**Included Methods:**
- **Bond-Boost** (BB): Parabolic bias with envelope function that vanishes near TS
- **MMF Simple**: Ridge-based bias using minimum-mode following
- **J-MMF Shear**: Jacobian-propagated climbing direction with orthogonal projection

**Included Files:**
- `HDkit/` — Core HD calculator library (Bond-Boost, MMF, BasinManager)
- `run_hd.py` — Unified multi-step HD-MD runner
- `run_compare.py` — Single-step bias comparison across all methods
- `verify.py` — Environment verification script
- `F19.traj` / `F1.traj` — Initial structures
- `Cu_u3.eam` — Cu EAM potential file
- `README.md` / `README.zh-CN.md` — English & Chinese documentation

**Requirements:**
- Python ≥ 3.10
- ASE ≥ 3.22
- LAMMPS with Python bindings
- Linux or macOS (Windows via WSL)