#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: Qian Lixiang
# @Email: 649811459@qq.com
__author__ = "Qian Lixiang"
__email__ = "649811459@qq.com"

"""
Bond-Boost Hyperdynamics Calculation Module

This module implements the Bond-Boost Hyperdynamics method, which accelerates
rare event sampling by monitoring atomic bond strains and applying a bias
potential. Compared to the MMF method, Bond-Boost does not require Hessian
computation, making it more computationally efficient and suitable for
large-scale systems and high-temperature simulations.

Core Principles:
1. Identify the basin and record initial bond lengths
2. During MD, compute the strain of all bonds ε_i = (r_i - r_i^0) / r_i^0
3. Find the maximum strain bond ε_m
4. Apply a bias potential E_bias = A(ε_m) * Σ_i δV_i(ε_i)
   - δV_i: per-bond parabolic bias potential
   - A(ε_m): envelope function that ensures the bias potential gradually
     reduces to zero when approaching the transition state

Typical usage:
    >>> from ase.calculators.emt import EMT
    >>> from ase import Atoms
    >>> from ase.md import VelocityVerlet
    >>> 
    >>> # Create a Bond-Boost calculator
    >>> bb_calc = BondBoostCalculator(
    >>>     std_calc=EMT(),
    >>>     emax=0.5,  # Maximum bias energy (eV)
    >>>     cutoff=3.0,  # Bond distance cutoff (Å)
    >>>     temperature_K=300
    >>> )
    >>> 
    >>> atoms = Atoms('Cu64', ...)
    >>> atoms.calc = bb_calc
    >>> 
    >>> # Run hyperdynamics MD
    >>> dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
    >>> dyn.run(10000)
"""

from typing import List, Optional, Any, Union
from ase import Atoms, units
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from ase.io import Trajectory
import numpy as np

from .basecalculator import BaseCalculator
from ..basin import BasinManager


class BondBoostCalculator(BaseCalculator):
    """
    Bond-Boost Hyperdynamics Bias Potential Calculator
    
    Accelerates rare event sampling by monitoring atomic bond strains and
    applying a bias potential. Compared to the MMF method, Bond-Boost does
    not require Hessian matrix computation, making it more computationally
    efficient and particularly suitable for large-scale systems and
    high-temperature simulations.
    
    Algorithm flow:
    1. **Basin identification**: use BasinManager to identify and record the
       basin structure
    2. **Bond topology construction**: establish bond connectivity based on
       the cutoff parameter
    3. **Strain calculation**: compute strain for each bond
       ε_i = (r_i - r_i^0) / r_i^0
    4. **Bias potential calculation**:
       - Per-bond bias: δV_i = (emax/Nb) * (1 - (ε_i/q)^2) if |ε_i| < q, else 0
       - Total bias base: V_b = Σ_i δV_i
       - Envelope function: A(ε_m) = (1-(ε_m/q)^2)^2 / (1-P1^2(ε_m/q)^2)
       - Total bias potential: E_bias = A(ε_m) * V_b
    5. **Force calculation**: compute bias forces via the chain rule
       F_bias = -∇E_bias
    
    Parameters
    ----------
    std_calc : Calculator
        Standard PES calculator (e.g., EMT, VASP)
    
    logfile : str or file object, default='Bond.log'
        Log output file
    
    cutoff : float, default=3.0
        Bond distance cutoff (Å). Atoms closer than cutoff are considered
        bonded.
        - Too small: may miss bonds, leading to insufficient bias
        - Too large: includes excessive non-bonded interactions, reducing
          efficiency
        Recommended: set to the second-nearest-neighbor distance
    
    emax : float, default=0.5
        Maximum bias energy (eV). Controls the strength of the bias potential.
        - Too small: acceleration effect is not obvious
        - Too large: may introduce non-physical behavior
        Recommended range: 0.3 ~ 1.0 eV
    
    q : float, default=0.37
        Strain cutoff parameter. When |ε_i| > q, the bias potential for that
        bond is zero.
        - Smaller value: bias potential begins to decay at smaller strain
          (conservative)
        - Larger value: allows larger strain (aggressive)
        Recommended range: 0.3 ~ 0.5
    
    max_q : float, default=2.0
        Maximum strain threshold multiplier. When ε_m > max_q*q, the system
        is considered to have left the basin and re-identification is needed.
    
    P1 : float, default=0.9
        Envelope function parameter. Controls the decay rate of the envelope
        function as ε_m approaches q.
        - Near 1: slow decay
        - Near 0: fast decay
        Recommended range: 0.8 ~ 0.95
    
    delta : float, default=1e-6
        Numerical differentiation step size (reserved parameter, currently
        unused)
    
    temperature_K : float, optional
        System temperature (K). Used to compute the boost factor
        ACT = exp(E_bias/kT).
    
    write_basins : bool, default=True
        Whether to write newly discovered basins to 'basins.traj'
    
    write_bias_log : bool, default=True
        Whether to write bias energy, temperature, and ACT to 'bias.log'
    
    bias_interval : int, default=10
        bias.log output interval (output every N calculate() calls)
    
    pbc_wrap : bool, default=True
        Whether to wrap atoms into the primary cell after each calculate()
        call (reserved parameter, currently unused)
    
    Attributes
    ----------
    bias_energy : float
        Current bias energy (eV)
    
    bias_forces : np.ndarray, shape (n_atoms, 3)
        Current bias forces (eV/Å)
    
    basin_id : int
        ID of the current basin
    
    n_bonds : int
        Number of bonds in the current basin
    
    epsilon_m : float
        Maximum bond strain
    
    Examples
    --------
    Basic usage:
    
    >>> from ase.build import bulk
    >>> from ase.calculators.emt import EMT
    >>> from ase.md import VelocityVerlet
    >>> from ase import units
    >>> 
    >>> # Create atomic structure
    >>> atoms = bulk('Cu', 'fcc', a=3.6).repeat((4, 4, 4))
    >>> atoms.rattle(stdev=0.05)
    >>> 
    >>> # Create Bond-Boost calculator
    >>> bb_calc = BondBoostCalculator(
    >>>     std_calc=EMT(),
    >>>     emax=0.5,
    >>>     cutoff=3.2,
    >>>     q=0.37,
    >>>     temperature_K=300
    >>> )
    >>> 
    >>> atoms.calc = bb_calc
    >>> 
    >>> # Run hyperdynamics MD
    >>> dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
    >>> for i in range(100):
    >>>     dyn.run(100)
    >>>     bias_E = bb_calc.parameters['bias_energy']
    >>>     eps_m = bb_calc.epsilon_m
    >>>     print(f"Step {i*100}: Bias={bias_E:.3f} eV, ε_m={eps_m:.4f}")
    
    Adjusting parameters for different systems:
    
    >>> # High-temperature system, increase emax
    >>> bb_calc = BondBoostCalculator(
    >>>     std_calc=EMT(),
    >>>     emax=0.8,
    >>>     q=0.4,
    >>>     temperature_K=600
    >>> )
    >>> 
    >>> # Low-temperature system, decrease emax
    >>> bb_calc = BondBoostCalculator(
    >>>     std_calc=EMT(),
    >>>     emax=0.3,
    >>>     q=0.3,
    >>>     temperature_K=100
    >>> )
    
    Notes
    -----
    - The energy and forces returned by calculate() already include the bias
      terms
    - Bond topology is established once per basin and remains unchanged during
      the MD run
    - When max_q*q < ε_m, the system automatically attempts to identify a new
      basin and update the bond topology
    - Bond length calculations automatically account for periodic boundary
      conditions (minimum image convention)
    - Bias forces are computed exactly via the chain rule, not via numerical
      differentiation
    - Detailed output from BasinManager is written to 'rlx.log'
    - The Bond-Boost method does not guarantee time reversibility, so it is
      not suitable for equilibrium property calculations
    
    References
    ----------
    .. [1] Miron, R. A., & Fichthorn, K. A. (2003). Accelerated molecular 
           dynamics with the bond-boost method. The Journal of Chemical 
           Physics, 119(12), 6210-6216.
    .. [2] Voter, A. F., Montalenti, F., & Germann, T. C. (2002). Extending 
           the time scale in atomistic simulation of materials. Annual Review 
           of Materials Research, 32(1), 321-346.
    """

    default_parameters = {
        # Results storage
        "basin_id": -1,
        "bias_energy": None,
        "bias_forces": None,
        "fcalls": 0,
    }

    def __init__(
        self,
        std_calc: Calculator,
        # Calculator parameters
        cutoff: float = 3.0,
        emax: float = 0.5,
        q: float = 0.37,
        max_q: float = 2.0,
        P1: float = 0.9,
        # Other parameters
        temperature_K: Optional[float] = None,
        write_basins: bool = True,
        write_bias_log: bool = True,
        logfile: Optional[Union[str, object]] = "Bond.log",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            std_calc = std_calc,
            logfile = logfile,
            **kwargs,
        )

        self._cutoff = cutoff
        self.emax = emax
        self._q = q
        self._max_q = max_q
        self._P1 = P1
        self._temperature_K = temperature_K

        self.bm = BasinManager(verbose=verbose)
        self._verbose = verbose

        class DummyTrajectoryWriter:
            def write(self, atoms, **kwargs):
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
        self.bias_log = (
            self._exit_stack.enter_context(open("bias.log", "a", encoding="utf-8"))
            if write_bias_log
            else DummyLogWriter()
        )

        # Basin-related parameters (updated once per basin)
        self.positions0: Optional[np.ndarray] = None  # basin positions (n_atoms, 3)
        self.bond_indices: np.ndarray  # bond indices (n_bonds, 2)
        self.n_bonds: int = 0  # number of bonds
        self.bond_lengths0: np.ndarray  # basin bond lengths (n_bonds,)

        # Parameters updated during MD
        self.positions: np.ndarray  # current positions (n_atoms, 3)
        self.bond_lengths: np.ndarray  # current bond lengths (n_bonds,)
        self.bond_vectors: np.ndarray  # bond vectors (n_bonds, 3)
        self.epsilons: np.ndarray  # bond strains (n_bonds,)
        self.grad_epsilons_per_bond: np.ndarray  # strain gradients (n_bonds, 3)
        self.delta_potentials: np.ndarray  # per-bond bias potentials (n_bonds,)
        self.Vb: float  # total bias potential base
        self.grad_delta_potentials: np.ndarray  # bias potential gradients (n_bonds,)
        self.epsilon_m_index: int  # maximum strain bond index
        self.epsilon_m: float = 0.0  # maximum strain
        self.grad_epsilon_m: np.ndarray  # maximum strain gradient (3,)
        self.A: float  # envelope function value
        self.grad_A: float  # envelope function gradient
        self.bias_energy: float  # bias energy
        self.bias_forces: np.ndarray  # bias forces (n_atoms, 3)

    def _calc_pbc_bond_vectors(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute bond vectors accounting for periodic boundary conditions
        
        Uses the minimum image convention to compute bond vectors, ensuring
        the shortest bond distance is obtained.
        
        Parameters
        ----------
        positions : np.ndarray, shape (n_atoms, 3)
            Atomic positions
        
        Returns
        -------
        bond_vectors : np.ndarray, shape (n_bonds, 3)
            Bond vectors
        """
        p1 = positions[self.bond_indices[:, 0]]
        p2 = positions[self.bond_indices[:, 1]]
        diff = p1 - p2

        pbc = self.atoms.get_pbc()
        if not np.any(pbc):
            return diff

        cell = self.atoms.get_cell()
        if np.abs(np.linalg.det(cell)) < 1e-10:
            return diff

        # Convert to fractional coordinates
        cell_inv = np.linalg.inv(cell)
        scaled_diff = diff @ cell_inv

        # Apply minimum image convention
        for i in range(3):
            if pbc[i]:
                scaled_diff[:, i] = scaled_diff[:, i] - np.round(
                    scaled_diff[:, i]
                )

        return scaled_diff @ cell

    def _calc_delta_potentials(self, epsilons: np.ndarray) -> np.ndarray:
        """
        Compute per-bond bias potentials
        
        δV_i = (emax/Nb) * (1 - (ε_i/q)^2) if |ε_i| < q, else 0
        
        Parameters
        ----------
        epsilons : np.ndarray, shape (n_bonds,)
            Bond strains
        
        Returns
        -------
        delta_potentials : np.ndarray, shape (n_bonds,)
            Per-bond bias potentials
        """
        emax = self.emax
        q = self._q
        Nb = self.n_bonds
        results = emax * (1 - (epsilons / q) ** 2) / Nb
        results[np.abs(epsilons) > q] = 0
        return results

    def _calc_grad_delta_potential(self, epsilons: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of per-bond bias potentials w.r.t. strain
        
        ∂(δV_i)/∂ε_i = -2*emax*ε_i / (Nb*q^2) if |ε_i| < q, else 0
        
        Parameters
        ----------
        epsilons : np.ndarray, shape (n_bonds,)
            Bond strains
        
        Returns
        -------
        gradients : np.ndarray, shape (n_bonds,)
            Bias potential gradients
        """
        emax = self.emax
        q = self._q
        Nb = self.n_bonds
        results = -2 * emax * epsilons / q**2 / Nb
        results[np.abs(epsilons) > q] = 0
        return results

    def _calc_envelope(self, epsilon_m: float) -> float:
        """
        Compute the envelope function
        
        A(ε_m) = (1-(ε_m/q)^2)^2 / (1-P1^2(ε_m/q)^2) if |ε_m| < q, else 0
        
        The envelope function ensures the bias potential smoothly decays to
        zero as the system approaches the transition state (ε_m near q).
        
        Parameters
        ----------
        epsilon_m : float
            Maximum strain
        
        Returns
        -------
        A : float
            Envelope function value
        """
        q = self._q
        P1 = self._P1
        if np.abs(epsilon_m) > q:
            return 0.0
        else:
            term = 1 - (epsilon_m / q) ** 2
            denom = 1 - P1**2 * (epsilon_m / q) ** 2
            return term * term / denom

    def _calc_grad_envelope(self, epsilon_m: float) -> float:
        """
        Compute the gradient of the envelope function w.r.t. maximum strain
        
        ∂A/∂ε_m
        
        Parameters
        ----------
        epsilon_m : float
            Maximum strain
        
        Returns
        -------
        dA : float
            Envelope function gradient
        """
        q = self._q
        P1 = self._P1
        x = epsilon_m / q
        if np.abs(epsilon_m) > q:
            return 0.0
        else:
            numerator = -2 * x * (1 - x**2) * (2 - P1**2 - P1**2 * x**2)
            denominator = q * (1 - P1**2 * x**2) ** 2
            return numerator / denominator

    def _update_state0(self, sid: int):
        """Refresh basin reference: positions, bond topology, bond lengths.

        Called after ``bm.map_atoms_to_basin`` identifies a new basin.
        """
        self._log(f"# Updating basin reference: basin_id = {sid}")
        self.positions0 = self.bm.known_minima_positions[sid].reshape(-1, 3)
        self.write_basins.write(self.bm.export_basin(sid))

        # Build bond topology from the *relaxed* basin positions
        n_atoms = len(self.positions0)
        atoms_ref = self.atoms.copy()
        atoms_ref.set_positions(self.positions0)
        nl = NeighborList(
            [self._cutoff / 2] * n_atoms,
            skin=0.0,
            self_interaction=False,
            bothways=True,
        )
        nl.update(atoms_ref)
        neighbor_list = nl.nl.neighbors

        bond_set = set()
        for i in range(len(neighbor_list)):
            neighbors = neighbor_list[i]
            if neighbors is None:
                continue
            for j in neighbors:
                if i < j:
                    bond_set.add((i, j))

        self.bond_indices = np.array(list(bond_set), dtype=int)
        self.n_bonds = len(self.bond_indices)

        # Reference bond lengths (PBC-aware)
        bond_vectors0 = self._calc_pbc_bond_vectors(self.positions0)
        self.bond_lengths0 = np.linalg.norm(bond_vectors0, axis=1)

        self._log(f"# Basin {sid}: {self.n_bonds} bonds identified")

    def _update_positions(self):
        """Update current atomic positions"""
        self.positions = self.atoms.get_positions()

    def _update_bond_lengths(self):
        """Update current bond lengths"""
        self.bond_vectors = self._calc_pbc_bond_vectors(self.positions)
        self.bond_lengths = np.linalg.norm(self.bond_vectors, axis=1)

    def _update_epsilons(self):
        """Update bond strains ε_i = (r_i - r_i^0) / r_i^0"""
        self.epsilons = (self.bond_lengths - self.bond_lengths0) / self.bond_lengths0

    def _update_grad_epsilons(self):
        """
        Update strain gradients ∂ε_i/∂r_α
        
        For bond i connecting atoms (a1, a2):
        ∂ε_i/∂r_a1 = (r_a1 - r_a2) / (r_i^0 * r_i)
        ∂ε_i/∂r_a2 = -∂ε_i/∂r_a1
        """
        diff = self.bond_vectors  # (n_bonds, 3)
        prefactor = 1.0 / (self.bond_lengths0 * self.bond_lengths)
        prefactor = prefactor[:, np.newaxis]
        self.grad_epsilons_per_bond = diff * prefactor  # (n_bonds, 3)

    def _update_delta_potentials(self):
        """Update per-bond bias potentials and total bias potential base"""
        self.delta_potentials = self._calc_delta_potentials(self.epsilons)
        self.Vb = self.delta_potentials.sum()

    def _update_grad_delta_potentials(self):
        """Update per-bond bias potential gradients"""
        self.grad_delta_potentials = self._calc_grad_delta_potential(
            self.epsilons
        )

    def _update_epsilon_m(self):
        """
        Update maximum strain and its gradient
        
        ε_m = max_i |ε_i|
        ∂ε_m/∂r_α = sign(ε_m) * ∂ε_m/∂r_α
        """
        index = int(np.argmax(np.abs(self.epsilons)))
        self.epsilon_m_index = index

        raw_epsilon = self.epsilons[index]
        self.epsilon_m = np.abs(raw_epsilon)
        grad_vec = self.grad_epsilons_per_bond[index]
        sign = np.sign(raw_epsilon)

        self.grad_epsilon_m = sign * grad_vec

    def _update_envelope(self):
        """Update envelope function value"""
        self.A = self._calc_envelope(self.epsilon_m)

    def _update_grad_envelope(self):
        """Update envelope function gradient"""
        self.grad_A = self._calc_grad_envelope(self.epsilon_m)

    def _update_bias(self):
        """
        Update bias potential and bias forces
        
        Bias energy: E_bias = A(ε_m) * V_b
        Bias forces: F_bias = -∇E_bias (computed via the chain rule)
        """
        self._update_positions()
        self._update_bond_lengths()
        self._update_epsilons()
        self._update_grad_epsilons()
        self._update_delta_potentials()
        self._update_grad_delta_potentials()
        self._update_epsilon_m()
        self._update_envelope()
        self._update_grad_envelope()

        # Bias energy
        self.bias_energy = self.A * self.Vb

        # Bias force calculation: F_bias = -A * Σ_i (∂δV_i/∂ε_i) * (∂ε_i/∂r)
        #                      - V_b * (∂A/∂ε_m) * (∂ε_m/∂r)
        coeffs = self.A * self.grad_delta_potentials  # (n_bonds,)

        m_idx = self.epsilon_m_index
        raw_eps_m = self.epsilons[m_idx]
        sign_m = np.sign(raw_eps_m)
        term2_coeff = self.Vb * self.grad_A * sign_m
        coeffs[m_idx] += term2_coeff

        force_contributions = (
            -1.0 * coeffs[:, np.newaxis] * self.grad_epsilons_per_bond
        )

        self.bias_forces = np.zeros((self.n_atoms, 3))
        np.add.at(
            self.bias_forces, self.bond_indices[:, 0], force_contributions
        )
        np.add.at(
            self.bias_forces, self.bond_indices[:, 1], -force_contributions
        )

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: List[str] = None,
        system_changes: List[str] = all_changes,
    ):
        """
        Compute Bond-Boost bias energy and forces.

        Workflow
        --------
        1. Update the basin reference if needed (first call or large strain).
        2. Compute bond strains and the bias potential.
        3. Return total energy = E_std + bias_energy,
           total forces = F_std + bias_forces.

        A new basin is identified automatically when
        :math:`\\varepsilon_m > \\mathrm{max\\_q} \\cdot q`.
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        if properties is None:
            properties = self.implemented_properties

        self.n_atoms = self.atoms.get_global_number_of_atoms()

        # Unbiased energy & forces
        self._update_std_results()

        # ── Check if the basin reference needs updating ──
        need_update = self.positions0 is None
        if self.epsilon_m > self._max_q * self._q:
            need_update = True

        if need_update:
            atoms_tmp = self.atoms.copy()
            atoms_tmp.calc = self.std_calc
            sid = self.bm.map_atoms_to_basin(atoms_tmp)

            if sid is not None:
                if sid != self.parameters["basin_id"]:
                    self.parameters["basin_id"] = sid
                    self._update_state0(sid)
            else:
                if self.positions0 is not None:
                    self._log(
                        "# Warning: Unable to identify new stable state. "
                        "Keeping previous state."
                    )
                else:
                    self._log("# Error: Unable to identify stable state")
                    raise RuntimeError(
                        "BondBoost: Unable to identify stable state"
                    )

        # ── Compute bias ──
        self._update_bias()

        bias_energy = self.bias_energy
        self.parameters["bias_energy"] = bias_energy
        self.results["energy"] = self.std_results["energy"] + bias_energy

        bias_forces = self.bias_forces
        self.parameters["bias_forces"] = bias_forces.copy()
        self.results["forces"] = self.std_results["forces"] + bias_forces

        self.fcalls += self.bm.fcalls
        self.bm.fcalls = 0
        self.parameters["fcalls"] = self.fcalls

        # ── Write bias.log ──
        if self.bias_log.tell() == 0:
            self.bias_log.write("# bias_energy    temperature    ACT\n")

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