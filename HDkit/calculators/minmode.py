#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: Qian Lixiang
# @Email: 649811459@qq.com
__author__ = "Qian Lixiang"
__email__ = "649811459@qq.com"

"""
Hessian Minimum Mode Calculation Module

This module provides the MinModeCalculator class for computing the minimum eigenvalue
and eigenvector of the Hessian matrix of an atomic system. Primarily used to identify
escape directions on the potential energy surface (PES), serving as a core component
of hyperdynamics methods (e.g., ridge-based).

Supports two calculation methods:
1. Lanczos iteration: efficient, suitable for large-scale systems, supports warm-start
   to accelerate convergence
2. Vibration full-Hessian method: accurate, suitable for small-scale systems or as a
   fallback when Lanczos fails to converge

Typical usage:
    >>> from ase.calculators.emt import EMT
    >>> from ase import Atoms
    >>> 
    >>> atoms = Atoms('Cu4', ...)
    >>> atoms.calc = EMT()
    >>> 
    >>> # Use the Lanczos method to compute the minimum mode
    >>> mmcalc = MinModeCalculator(std_calc=EMT(), algo='Lanczos')
    >>> atoms.calc = mmcalc
    >>> forces = atoms.get_forces()  # Returns the minimum mode direction
    >>> 
    >>> # Get the eigenvalue
    >>> eig_val = mmcalc.parameters['eig_values'][0]
    >>> print(f"Minimum eigenvalue: {eig_val:.6f}")
"""

from typing import List, Optional, Tuple, Union
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
import numpy as np
import gc

from .basecalculator import BaseCalculator

def vunit(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector
    
    Parameters
    ----------
    v : np.ndarray
        Input vector
    
    Returns
    -------
    normalized : np.ndarray
        The normalized vector. If the input is a zero vector, the original
        vector is returned.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def vectors_angles(v1: np.ndarray, v2: np.ndarray, minimum: bool = True) -> np.ndarray:
    """
    Compute the angles between two sets of vectors
    
    Supports angle calculation for single vectors or vector sets,
    returning angles in degrees.
    
    Parameters
    ----------
    v1, v2 : np.ndarray
        Input vector(s). Shape can be:
        - 1D: (n,) single vector
        - 2D: (n, m) m vectors of dimension n
    minimum : bool, default=True
        Whether to return the minimum angle (considering vector reversal).
        - True: return min(angle, 180-angle)
        - False: return raw angle [0, 180]
    
    Returns
    -------
    angles : np.ndarray
        Angle array (degrees)
    
    Raises
    ------
    ValueError
        If the two input shapes do not match
    
    Examples
    --------
    >>> v1 = np.array([1, 0, 0])
    >>> v2 = np.array([0, 1, 0])
    >>> angle = vectors_angles(v1, v2)
    >>> print(f"Angle: {angle:.1f} degrees")  # 90.0 degrees
    """
    a = np.atleast_2d(v1.T).T if np.ndim(v1) == 1 else np.asarray(v1)
    b = np.atleast_2d(v2.T).T if np.ndim(v2) == 1 else np.asarray(v2)
    
    if a.shape != b.shape:
        raise ValueError(f"Input shape mismatch: {a.shape} vs {b.shape}")
    
    dots = np.sum(a * b, axis=0)
    norms = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    
    # Safely handle zero-vector case
    cos_theta = np.zeros_like(dots)
    with np.errstate(invalid="ignore"):
        np.divide(dots, norms, out=cos_theta, where=(norms != 0))
    
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.arccos(cos_theta) * 180 / np.pi
    
    if minimum:
        return np.minimum(angles, 180 - angles)
    else:
        return angles

class MinModeCalculator(BaseCalculator):
    """
    Hessian Minimum Mode Calculator
    
    Computes the minimum eigenvalue and corresponding eigenvector of the
    Hessian matrix of an atomic system, used to identify the "easiest escape"
    direction on the PES. This is a core component of Min-Mode Following and
    other hyperdynamics methods.
    
    Supports two computation algorithms:
    
    1. **Lanczos Iteration** (recommended)
       - Principle: builds a tridiagonal matrix via Lanczos iteration and
         solves for its minimum eigenvalue
       - Advantages: computationally efficient, low memory footprint,
         suitable for large-scale systems
       - Supports warm-start: can use the previous step's minimum mode as
         the initial vector to accelerate convergence
       - Convergence criterion: eigenvector rotation angle < tolerance
    
    2. **Vibration Full-Hessian Method** (fallback)
       - Principle: computes the full Hessian matrix via finite differences
         and diagonalizes it
       - Advantages: accurate, does not depend on iterative convergence
       - Disadvantages: high computational cost and memory usage
       - Use case: small systems or when Lanczos fails to converge
    
    Automatic fallback mechanism:
    - If Lanczos does not converge, automatically retries from a random vector
    - If the random vector also fails, automatically switches to the Vibration method
    
    Parameters
    ----------
    std_calc : Calculator
        Standard PES calculator (e.g., EMT, VASP) for computing energy and forces
    
    algo : str, default='Lanczos'
        Minimum-mode calculation algorithm:
        - 'Lanczos', 'lan', 'l': Lanczos iteration
        - 'Vibration', 'vib', 'v': full-Hessian diagonalization
    
    logfile : str or file object, default='mode.log'
        Log output file. Set to None to use os.devnull and disable logging.
    
    n_eigs : int, default=1
        Number of smallest eigenvalues to compute. Usually 1 is sufficient.
        Note: n_eigs must not exceed the number of degrees of freedom of
        unconstrained atoms (3*n_unconstrained_atoms)
    
    indices : list of int, optional
        List of atom indices participating in the calculation. If None,
        automatically selects all unconstrained atoms. Used to reduce
        computational cost, e.g., by considering only surface atoms.
    
    delta : float, default=1e-4
        Finite-difference displacement step size (Å). Affects Hessian
        calculation accuracy.
        - Too large: insufficient precision
        - Too small: increased numerical error
        Recommended range: 1e-5 ~ 1e-3
    
    tolerance : float, default=1e-2
        Lanczos convergence tolerance (degrees). Convergence is considered
        reached when the rotation angle of eigenvectors between successive
        iterations falls below this value.
        - Smaller: more accurate, but requires more iterations
        - Larger: faster, but lower precision
        Recommended range: 1e-3 ~ 1e-1
    
    direction : {'central', 'forward', 'backward'}, default='central'
        Finite-difference scheme:
        - 'central': (f(x+δ) - f(x-δ)) / 2δ, high precision but doubles cost
        - 'forward': (f(x+δ) - f(x)) / δ
        - 'backward': (f(x) - f(x-δ)) / δ
    
    max_niter : int, default=200
        Maximum Lanczos iterations. If not converged, retry or switch algorithm.
    
    min_mode : np.ndarray, optional
        Minimum-mode vector from the previous step, used for Lanczos warm-start.
        - None: start from a random vector
        - not None: start from this vector (accelerates convergence)
        Set via calc.parameters['min_mode'] = vector
    
    Attributes
    ----------
    fcalls : int
        Cumulative number of force evaluations, for performance monitoring
    
    Examples
    --------
    Basic usage:
    
    >>> from ase.build import bulk
    >>> from ase.calculators.emt import EMT
    >>> 
    >>> # Create atomic structure
    >>> atoms = bulk('Cu', 'fcc', a=3.6)
    >>> atoms.rattle(stdev=0.1)
    >>> 
    >>> # Create minimum-mode calculator
    >>> mmcalc = MinModeCalculator(
    >>>     std_calc=EMT(),
    >>>     algo='Lanczos',
    >>>     tolerance=1e-2,
    >>>     logfile='mode.log'
    >>> )
    >>> 
    >>> # Compute the minimum mode
    >>> atoms.calc = mmcalc
    >>> min_mode_forces = atoms.get_forces()  # Returns minimum-mode direction vector
    >>> 
    >>> # Get the eigenvalue
    >>> eig_val = mmcalc.parameters['eig_values'][0]
    >>> print(f"Minimum eigenvalue: {eig_val:.6f}")
    >>> 
    >>> # Check if stable state (eig_val > 0) or saddle point (eig_val < 0)
    >>> if eig_val > 0:
    >>>     print("This is a stable minimum")
    >>> else:
    >>>     print("This is a saddle point")
    
    Warm-start for accelerated convergence:
    
    >>> # Step 1: normal calculation
    >>> mmcalc.parameters['min_mode'] = None
    >>> atoms.calc = mmcalc
    >>> _ = atoms.get_forces()
    >>> 
    >>> # Save the minimum mode
    >>> saved_mode = mmcalc.parameters['min_mode']
    >>> 
    >>> # Step 2: warm-start using the previous result
    >>> atoms2 = atoms.copy()
    >>> atoms2.rattle(stdev=0.05)
    >>> mmcalc.parameters['min_mode'] = saved_mode
    >>> atoms2.calc = mmcalc
    >>> _ = atoms2.get_forces()  # Converges faster
    
    Computing multiple smallest eigenvalues:
    
    >>> mmcalc = MinModeCalculator(
    >>>     std_calc=EMT(),
    >>>     n_eigs=3  # Compute 3 smallest eigenvalues
    >>> )
    >>> atoms.calc = mmcalc
    >>> _ = atoms.get_forces()
    >>> eig_vals = mmcalc.parameters['eig_values']
    >>> print(f"3 smallest eigenvalues: {eig_vals}")
    
    Notes
    -----
    - The forces returned by calculate() are the minimum-mode direction
      vector (normalized), not real forces
    - If F0·min_mode < 0, return min_mode; otherwise return -min_mode
      (to ensure alignment with the force direction)
    - Hessian computation uses finite differences; cost scales with
      number of atoms and delta
    - For large systems (>100 atoms), the Lanczos method is strongly recommended
    - Memory optimization: large arrays are explicitly deleted after use
      and gc.collect() is called
    - Fully compatible with MPI-parallel calculators (e.g., VASP);
      no file caching is used
    
    References
    ----------
    .. [1] Henkelman, G., & Jónsson, H. (1999). A dimer method for finding 
           saddle points on high dimensional potential surfaces using only 
           first derivatives. The Journal of Chemical Physics, 111(15), 
           7010-7022.
    .. [2] Lanczos, C. (1950). An iteration method for the solution of the 
           eigenvalue problem of linear differential and integral operators.
    """

    default_parameters = {
        # Results storage
        "min_mode": None,                                       # (N, 3)
        "eig_values": np.array(None),                           # (n_eigs,)
        "eig_vectors": np.array(None),                          # (3N, n_eigs)
        "fcalls": 0,
    }

    def __init__(
        self,
        std_calc: Calculator,
        # Calculator settings
        algo: str = "Lanczos",                                  # minmode calculation algorithm
        n_eigs: int = 1,                                        # number of smallest eigenvalues to compute
        indices: Optional[List[int]] = None,                    # atom indices participating in the calculation
        delta: float = 1e-4,                                    # finite-difference step size
        # Lanczos settings
        tolerance: float = 1e-2,                                # Lanczos convergence tolerance (degrees)
        direction: str = "central",                             # finite-difference scheme
        orth: Optional[bool] = None,                            # whether to fully orthogonalize, None for auto
        max_niter: int = 100,                                   # Lanczos maximum iterations
        # Other settings
        logfile: Optional[Union[str, object]] = "mode.log",     # log file
        **kwargs,
    ):

        super().__init__(
            std_calc = std_calc,
            logfile = logfile,
            **kwargs,
        )

        self._algo = algo.lower()
        self._n_eigs = n_eigs
        self._indices = indices
        self._delta = delta
        self._tolerance = tolerance
        self._direction = direction
        self._orth = orth
        self._max_niter = max_niter

        self._atoms: Optional[Atoms] = None

    def _get_indices(self) -> List[int]:
        """
        Get the list of atom indices participating in the calculation
        
        If the user has not specified the indices parameter, automatically
        selects all unconstrained atoms.
        
        Returns
        -------
        indices : list of int
            List of atom indices participating in the calculation
        """
        if self._indices is None:
            self._update_indices()

        return self._indices

    def _update_indices(self) -> None:
        """
        Update the list of atom indices participating in the calculation
        """
        atoms = self.atoms.copy()
        n_atoms = len(atoms)

        # Automatically select unconstrained atoms
        unconstrained_indices = {
            i for i in range(n_atoms)
            if not any(i in c.index for c in atoms.constraints)
        }
        indices = list(unconstrained_indices)
        self._indices = indices

    def _get_mask(self) -> np.ndarray:
        """
        Get the atom mask
        
        Returns
        -------
        mask : np.ndarray, shape (n_atoms,), dtype=bool
            True indicates the atom participates in the calculation,
            False indicates it is constrained
        """
        mask = np.ones(self.n_atoms, bool)
        mask[self._get_indices()] = False
        return ~mask

    def _calculate_partial_forces(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian-vector product H*q
        
        Uses finite-difference approximation to compute df/dx, equivalent
        to the product of the Hessian matrix with vector q. This is the core
        operation of the Lanczos iteration, avoiding explicit construction
        of the full Hessian matrix.
        
        Parameters
        ----------
        q : np.ndarray, shape (3*n_atoms,)
            Direction vector (flattened)
        
        Returns
        -------
        H_q : np.ndarray, shape (3*n_atoms,)
            Hessian-vector product
        
        Notes
        -----
        - Components of constrained atoms are automatically set to 0
        - The finite-difference scheme is controlled by the direction parameter
        - Each call adds 1-2 force evaluations (depending on direction)
        """
        delta = self._delta
        mask = self._get_mask()
        q = q.reshape(-1, 3)
        q[~mask] = 0.0
        displacement = vunit(q)

        atoms = self.atoms.copy()
        atoms.calc = self.std_calc
        pos = atoms.get_positions()
        f0 = self.std_results["forces"].flatten()

        match self._direction:
            case "central":
                # Central difference: high precision but doubles cost
                atoms.set_positions(pos + displacement * delta)
                f_plus = atoms.get_forces().flatten()
                atoms.set_positions(pos - displacement * delta)
                f_minus = atoms.get_forces().flatten()
                self.fcalls += 2
                df = (f_minus - f_plus) / (2 * delta)
            case "forward":
                # Forward difference
                atoms.set_positions(pos + displacement * delta)
                f_plus = atoms.get_forces().flatten()
                self.fcalls += 1
                df = (f0 - f_plus) / delta
            case "backward":
                # Backward difference
                atoms.set_positions(pos - displacement * delta)
                f_minus = atoms.get_forces().flatten()
                self.fcalls += 1
                df = (f_minus - f0) / delta
            case _:
                raise ValueError(f"Unknown direction: {self._direction}")

        return df

    def _get_min_modes_lanczos(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute minimum modes using the Lanczos iteration method
        
        The Lanczos algorithm builds the Krylov subspace of the Hessian
        iteratively, projecting it onto a tridiagonal matrix to efficiently
        solve for the smallest eigenvalue. Compared to full diagonalization,
        the computational cost is greatly reduced.
        
        Algorithm flow:
        1. Initialize: start from min_mode (warm-start) or a random vector
        2. Lanczos iteration:
           - Compute v = H*q - beta*q_prev
           - Orthogonalize (optional)
           - Update the tridiagonal matrix T
           - Solve eigenvalues of T, check convergence
        3. Convergence criterion: eigenvector rotation angle < tolerance
        4. Failure handling: restart or switch to Vibration method
        
        Parameters
        ----------
        atoms : Atoms
            Atoms object
        n_eigs : int, default=1
            Number of smallest eigenvalues to compute
        
        Returns
        -------
        eig_values : np.ndarray, shape (n_eigs,)
            The n_eigs smallest eigenvalues
        eig_vectors : np.ndarray, shape (3*n_atoms, n_eigs)
            The corresponding eigenvectors
        
        Notes
        -----
        - Memory optimization: pre-allocate all large arrays, clean up
          temporary variables promptly
        - If convergence fails from a warm-start vector, automatically
          retries with a random vector
        - If the random vector also fails, falls back to the Vibration method
        - Orthogonalization is only enabled when n_eigs > 1 to avoid
          unnecessary overhead
        """
        self._log("=" * 10)
        self._log("Lanczos iteration begins")
        self._log("-" * 10)

        indices = self._get_indices()
        mask = self._get_mask()
        if self._orth is None:
            orth = True if self._n_eigs > 1 else False
        else:
            orth = self._orth
        max_niter = self._max_niter
        ndim = self.n_atoms * 3

        q_prev = np.zeros(ndim)
        q1 = self.parameters["min_mode"]

        # Initialize: random vector or warm-start
        if q1 is None:
            self._log("Starting Lanczos from random vector")
            q = q_prev.reshape(-1, 3)
            q[mask] = np.random.uniform(-1, 1, (len(indices), 3))
        else:
            self._log("Starting Lanczos from inherited min_mode")
            q1 = q1.reshape(-1, 3)
            q = q1.copy()
            q[~mask] = 0.0
        q = vunit(q.flatten())

        # Pre-allocate arrays
        Q = np.zeros((ndim, max_niter + 1))
        alpha = np.zeros(max_niter)
        beta = np.zeros(max_niter + 1)
        T = np.zeros((max_niter, max_niter))

        Q[:, 0] = q
        beta[0] = 0

        prev_V = None
        converged = False
        eig_values = None
        current_V = None
        angles = None

        # tol = self._tolerance

        head = " " * 10
        for i in range(self._n_eigs):
            head += f"{'d_ANGLE' + str(i + 1):>14s}"
        self._log(head)

        # Lanczos main loop
        for k in range(1, max_niter + 1):
            q = Q[:, k - 1]
            v = self._calculate_partial_forces(q)
            v -= beta[k - 1] * q_prev
            alpha_k = q.T @ v
            alpha[k - 1] = alpha_k

            # Orthogonalization (only when n_eigs > 1)
            if orth:
                v -= (Q[:, :k].T @ v) @ Q[:, :k].T
            else:
                v -= alpha_k * q

            beta_k = np.linalg.norm(v)
            beta[k] = beta_k

            if beta_k < 1e-12:
                self._log("Beta near zero, Lanczos terminates early")
                break

            q_next = v / beta_k
            Q[:, k] = q_next

            # Update tridiagonal matrix T
            T[k - 1, k - 1] = alpha_k
            if k > 1:
                T[k - 2, k - 1] = beta[k - 1]
                T[k - 1, k - 2] = beta[k - 1]

            # Check convergence (when k >= n_eigs)
            if k >= self._n_eigs:
                T_sub = T[:k, :k]
                eig_values, T_vectors = np.linalg.eigh(T_sub)
                current_V = Q[:, :k] @ T_vectors[:, :self._n_eigs]

                if prev_V is not None:
                    angles = vectors_angles(prev_V, current_V)
                    angles_str = "".join(f"{angle:>14.4E}" for angle in angles)
                    self._log(f"  {k:>6}: {angles_str}")
                    if np.max(angles[:self._n_eigs]) < self._tolerance:
                        converged = True
                        break
                prev_V = current_V.copy()

            q_prev = q.copy()
            q = q_next

        # Output eigenvalues
        eigvalues_str = "".join(f"{eig_value:>14.4E}" for eig_value in eig_values[:self._n_eigs])
        self._log(f"eig_vals: {eigvalues_str}")

        # Check if eigenvalues are abnormally large
        if abs(eig_values[0]) > 1e5:
            self._log("Eigenvalue too large, convergence failed")
            converged = False

        # Process results
        if converged:
            self._log(f"Lanczos converged after {k} iterations!")
            self._log("=" * 10)
            result_eig = eig_values[:self._n_eigs].copy()
            result_V = current_V[:, :self._n_eigs].copy()
            return result_eig, result_V
        else:
            self._log(f"Lanczos did not converge. Max angle: {max(angles):.4f} degrees")
            was_inherited = (q1 is not None)

            if was_inherited:
                # Warm-start failed, retry with random vector
                self._log("Restarting from random vector")
                self._log("=" * 10)
                self.parameters["min_mode"] = None
                return self._get_min_modes_lanczos()
            else:
                # Random vector also failed, fall back to Vibration method
                self._log("Random vector failed. Falling back to Vibration method")
                self._log("=" * 10)
                return self._get_min_modes_vib()

    def _get_min_modes_vib(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute minimum modes using full-Hessian diagonalization
        
        Computes the full Hessian matrix via finite differences, then
        diagonalizes it to obtain all eigenvalues and eigenvectors.
        Compared to the ASE Vibrations class, this implementation avoids
        file caching and is fully compatible with MPI-parallel calculators.
        
        Algorithm flow:
        1. Displace each unconstrained atom in each direction (x, y, z)
        2. Compute Hessian columns using central differences
        3. Enforce Hessian symmetry (averaging)
        4. Diagonalize to obtain eigenvalues and eigenvectors
        5. Expand eigenvectors to the full system (constrained atoms set to 0)
        
        Parameters
        ----------
        atoms : Atoms
            Atoms object
        n_eigs : int, default=1
            Number of smallest eigenvalues to compute
        
        Returns
        -------
        eig_values : np.ndarray, shape (n_eigs,)
            The n_eigs smallest eigenvalues
        eig_vectors : np.ndarray, shape (3*n_atoms, n_eigs)
            The corresponding eigenvectors (full system)
        
        Notes
        -----
        - Cost: 6 force evaluations per atom (3 directions × ± displacement)
        - Memory: O(n_unconstrained²)
        - Use case: small systems (<50 atoms) or when Lanczos fails to converge
        - Fully compatible with MPI calculators like VASP; no file caching
        """
        self._log("=" * 10)
        self._log("Vibration method begins")
        self._log("-" * 10)

        atoms = self.atoms.copy()
        atoms.calc = self.std_calc
        indices = self._get_indices()
        mask = self._get_mask()
        delta = self._delta
        n_indices = len(indices)

        # Get reference forces
        pos0 = atoms.get_positions()
        f0 = self.std_results["forces"].flatten()

        # Initialize Hessian matrix (unconstrained atoms only)
        n_dof = n_indices * 3
        hessian = np.zeros((n_dof, n_dof))

        # Compute each column of the Hessian matrix
        for i, atom_idx in enumerate(indices):
            for j in range(3):  # x, y, z
                # Positive displacement
                atoms.positions[atom_idx, j] = pos0[atom_idx, j] + delta
                f_plus = atoms.get_forces().flatten()
                self.fcalls += 1

                # Negative displacement
                atoms.positions[atom_idx, j] = pos0[atom_idx, j] - delta
                f_minus = atoms.get_forces().flatten()
                self.fcalls += 1

                # Restore position
                atoms.positions[atom_idx, j] = pos0[atom_idx, j]

                # Compute Hessian column (central difference)
                df_full = (f_minus - f_plus) / (2.0 * delta)
                df_reduced = df_full.reshape(-1, 3)[mask].flatten()
                hessian[:, i * 3 + j] = df_reduced

                if (i * 3 + j + 1) % 3 == 0:
                    self._log(f"Progress: {i+1}/{n_indices} atoms")

        # Symmetrize Hessian (improve numerical precision)
        hessian = 0.5 * (hessian + hessian.T)

        # Diagonalize
        self._log("Diagonalizing Hessian matrix...")
        eig_values, eig_vectors_2d = np.linalg.eigh(hessian)

        # Diagnostic information
        self._log(f"Hessian shape: {hessian.shape}")
        self._log(f"Condition number: {np.linalg.cond(hessian):.2e}")
        eigvalues_str = "".join(f"{eig_value:>14.4E}" for eig_value in eig_values[:min(3, self._n_eigs)])
        self._log(f"Smallest {min(3, self._n_eigs)} eigenvalues: {eigvalues_str}")

        # Normalize eigenvectors
        eig_vectors = []
        for i in range(self._n_eigs):
            eig_vectors.append(vunit(eig_vectors_2d[:, i]))
        eig_vectors = np.array(eig_vectors).T

        self._log("=" * 10)

        # Expand to full system (constrained atoms set to 0)
        n_atoms = len(atoms)
        n_atoms_dof = n_atoms * 3
        eig_vectors_full = np.zeros((n_atoms_dof, self._n_eigs))
        mask_3d = np.repeat(mask, 3)

        for i in range(self._n_eigs):
            vec_reduced = eig_vectors[:, i]
            eig_vectors_full[mask_3d, i] = vec_reduced

        atoms.calc = None
        return eig_values[:self._n_eigs], eig_vectors_full

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: List[str] = None,
        system_changes: List[str] = all_changes,
    ):
        """Compute the Hessian minimum-mode direction.

        Returns the lowest eigenvector as ``forces``, so that
        ``MinModeCalculator`` can be used as a drop-in ASE Calculator.

        ``results['energy']`` is the unbiased PES energy.
        ``results['forces']`` is the *normalised* minimum-mode direction
        (NOT real forces).  The sign is chosen so that
        :math:`F_0 \\cdot N \\!<\\! 0` (i.e. the mode points against the
        force).
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        if properties is None:
            properties = self.implemented_properties

        self.n_atoms = atoms.get_global_number_of_atoms()

        # unbiased energy
        self._update_std_results()

        # compute minimum eigenpair
        match self._algo:
            case "lanczos" | "lan" | "l":
                eig_values, eig_vectors = self._get_min_modes_lanczos()
            case "vibration" | "vib" | "v":
                eig_values, eig_vectors = self._get_min_modes_vib()
            case _:
                raise ValueError(
                    f"Unknown algorithm: {self._algo}. "
                    "Use 'Lanczos' or 'Vibration'."
                )

        # store results
        eig_vectors = np.atleast_2d(eig_vectors)
        self.parameters["min_mode"] = eig_vectors[:, 0].reshape(-1, 3).copy()
        self.parameters["eig_values"] = eig_values.copy()
        self.parameters["eig_vectors"] = eig_vectors.copy()

        # extract and normalise the lowest mode
        min_mode = eig_vectors[:, 0].reshape(-1, 3)
        min_mode = vunit(min_mode)

        # align with force direction
        self.results["energy"] = self.std_results["energy"]
        F0 = self.std_results["forces"]
        if np.vdot(F0, min_mode) < 0:
            self.results["forces"] = min_mode
        else:
            self.results["forces"] = -min_mode

        self.parameters["fcalls"] = self.fcalls

        gc.collect()