#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: Qian Lixiang
# @Email: 649811459@qq.com
__author__ = "Qian Lixiang"
__email__ = "649811459@qq.com"

"""
Basin (energy well) management module

This module provides the BasinManager class for efficient identification,
storage, and management of basins on the potential energy surface (PES) in
molecular dynamics simulations. Key features include:

- Mapping atomic configurations to their corresponding basins via
  structure optimization
- Intelligent caching of known basins for fast recognition of revisited
  structures
- Distance calculations respecting periodic boundary conditions (PBC)
- Automatic detection and rejection of saddle points
- Persistent storage/loading of basin databases

Typical usage:
    >>> from ase import Atoms
    >>> from ase.calculators.emt import EMT
    >>> 
    >>> # Create manager
    >>> bm = BasinManager(min_dist_threshold=1.0, fmax=0.01)
    >>>
    >>> # Map a structure to its basin ID
    >>> atoms = Atoms('Cu4', ...)
    >>> atoms.calc = EMT()
    >>> basin_id = bm.map_atoms_to_basin(atoms)
    >>>
    >>> # Export a known basin
    >>> if basin_id is not None:
    >>>     basin_atoms = bm.export_basin(basin_id)
"""

import pickle
import numpy as np
import os
from typing import List, Optional, Type
from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.calculators.singlepoint import SinglePointCalculator
from .calculators.minmode import MinModeCalculator

def wrap_atoms(atoms) -> Atoms:
    """
    Wrap atomic positions according to periodic boundary conditions (PBC).

    Parameters
    ----------
    atoms : Atoms
        The atoms object to be wrapped.

    Returns
    -------
    Atoms
        The wrapped atoms object.
    """
    pbc = atoms.get_pbc()
    if not any(pbc):
        return atoms
    cell = atoms.get_cell()
    if np.abs(np.linalg.det(cell)) < 1e-10:
        return atoms
    pos = atoms.get_positions()

    # Convert to fractional coordinates
    cell_inv = np.linalg.inv(cell)
    scaled = pos @ cell_inv

    # Wrap periodic directions
    for i in range(3):
        if pbc[i]:
            scaled[:, i] = scaled[:, i] % 1.0

    # Convert back to Cartesian coordinates
    pos_wrapped = scaled @ cell

    atoms.set_positions(pos_wrapped)
    return atoms


class BasinFoundException(Exception):
    """
    Exception used to signal that a basin has been hit during optimization.

    Raised when the optimizer discovers during iteration that the atomic
    structure has entered the attraction basin of a known stable state.

    Attributes:
        basin_id: The ID of the basin that was hit.
    """
    def __init__(self, basin_id: int):
        self.basin_id = basin_id
        super().__init__(f"Basin {basin_id} found")


class BasinManager:
    """
    Energy basin (energy well) manager

    This class maps atomic configurations to local minima (basins) on the
    PES via structure optimization and maintains a database of known basins.
    Smart caching and PBC-aware distance calculations enable efficient
    basin identification.

    Workflow:
    1. Accept an input structure (must have a calculator attached)
    2. Check whether it is close to a known basin (distance threshold)
    3. If not close, run structure optimization
    4. During optimization, continuously check whether the structure enters
       the attraction basin of a known stable state
    5. If optimization converges to a new position, compute the smallest
       Hessian eigenvalue to determine whether it is a basin or saddle point
    6. If eigenvalue > 0, register as a new basin; if < 0, classify as a
       saddle point and refuse registration

    Parameters
    ----------
    min_dist_threshold : float, default=1.0
        Distance threshold (in Å) for deciding whether two structures
        belong to the same basin. When the L2 norm of the atomic position
        difference between two structures is smaller than this value,
        they are considered to belong to the same basin.
        Recommended values depending on system size and sensitivity needs:
        - Small systems (< 100 atoms): 0.5–1.0 Å
        - Large systems (> 100 atoms): 1.0–2.0 Å

    logfile : str, default='rlx.log'
        Path to the optimization log file. Records detailed information
        for each call to map_atoms_to_basin:
        - Step: optimization step count
        - Min-D: distance to the nearest basin
        - Reason: termination reason (FAST = fast hit, SADDLE = saddle
          point, empty = normal convergence)
        - Basin: ID of the hit basin (–1 indicates a saddle point)
        - Energy: current potential energy
        - fmax: current maximum force component
        - EigVal: smallest Hessian eigenvalue (computed only for new basins)

    fmax : float, default=1e-2
        Force convergence criterion for structure optimization (in eV/Å).
        Optimization is considered converged when the maximum force
        component on any atom falls below this value.

    optimizer_class : Type[Optimizer], optional
        The ASE optimizer class to use. Defaults to FIRE2.
        Other options: BFGSLineSearch, LBFGS, GPMin, etc.

    storage_file : str, default='basin.pkl'
        Path to the persistent storage file. Saves/loads the known-basin
        database, including:
        - Basin positions (PBC-wrapped)
        - Basin energies and forces
        - System topology (symbols, cell, PBC)
        If the file exists, it is loaded automatically at initialization.

    verbose : bool, default=False
        Whether to output detailed logs.
        - True: log every optimization step to the log file
        - False: log only the final result

    Attributes
    ----------
    n_basins : int
        Number of basins currently registered

    known_minima_positions : List[np.ndarray]
        List of atomic positions for each known basin;
        each element has shape (N_atoms*3,)

    known_minima_energies : List[float]
        List of energies for each known basin

    known_minima_forces : List[np.ndarray]
        List of force arrays for each known basin;
        each element has shape (N_atoms, 3)

    atoms : Atoms
        Reference atoms object that preserves the system's topology
        (symbols, cell, PBC)

    Examples
    --------
    Basic usage:

    >>> from ase.build import bulk
    >>> from ase.calculators.emt import EMT
    >>> 
    >>> # Create manager
    >>> bm = BasinManager(min_dist_threshold=0.8, fmax=0.01, verbose=True)
    >>>
    >>> # Create structure and attach calculator
    >>> atoms = bulk('Cu', 'fcc', a=3.6).repeat((2, 2, 2))
    >>> atoms.rattle(stdev=0.1)  # random perturbation
    >>> atoms.calc = EMT()
    >>>
    >>> # Identify basin
    >>> basin_id = bm.map_atoms_to_basin(atoms)
    >>> print(f"Structure mapped to basin {basin_id}")
    >>>
    >>> # Export basin structure
    >>> if basin_id is not None:
    >>>     basin_atoms = bm.export_basin(basin_id)
    >>>     print(f"Basin energy: {basin_atoms.get_potential_energy():.4f} eV")
    >>>
    >>> # Save database for later use
    >>> bm.save_to_file("my_basins.pkl")

    Loading from a saved database:

    >>> # Auto-load on next run
    >>> bm = BasinManager(storage_file="my_basins.pkl")
    >>> print(f"Loaded {bm.n_basins} known basins")

    Notes
    -----
    - This class automatically handles PBC using the minimum-image convention
      for distance calculations.
    - The optimizer instance is reused for efficiency to avoid repeated
      initialization overhead.
    - For large systems, consider increasing min_dist_threshold and fmax
      to speed up computation.
    - Saddle-point detection relies on Hessian smallest-eigenvalue
      calculation, which is relatively expensive and is performed only when
      optimization converges to a new position.
    """

    def __init__(
        self,
        min_dist_threshold: float = 1.0,
        logfile: str = "rlx.log",
        fmax: float = 1e-2,
        optimizer_class: Type[Optimizer] = None,
        storage_file: str = "basin.pkl",
        verbose: bool = False,
    ):

        # ===== Configuration parameters =====
        self._min_dist_threshold = min_dist_threshold
        self._logfile = logfile
        self._fmax = fmax
        self._storage_file = storage_file
        self._verbose = verbose
        
        if optimizer_class is None:
            # from ase.optimize import BFGSLineSearch
            # self._optimizer_class = BFGSLineSearch
            # from ase.optimize import LBFGS
            # self._optimizer_class = LBFGS
            from ase.optimize import FIRE2
            self._optimizer_class = FIRE2
        else:
            self._optimizer_class = optimizer_class

        # ===== Internal utilities =====
        self._mm_calc: MinModeCalculator = None  # For computing smallest Hessian eigenvalue
        self._opt_atoms: Optional[Atoms] = None  # Atoms object bound to the optimizer
        self._optimizer: Optional[Optimizer] = None  # Reusable optimizer instance
        self.fcalls: int = 0  # Force-call counter

        # ===== Basin database =====
        self.atoms: Optional[Atoms] = None  # Reference atoms object (no calc)
        self.known_minima_positions: List[np.ndarray] = []  # Basin position list
        self.known_minima_energies: List[float] = []  # Basin energy list
        self.known_minima_forces: List[np.ndarray] = []  # Basin force list
        self.n_basins = 0  # Number of registered basins

        # ===== Performance optimization cache =====
        self._dirty_centers = True  # Cache invalidation flag
        self._cached_centers_array: Optional[np.ndarray] = None  # Cached basin positions (Nb, 3N)

        # ===== Load existing data from file (if present) =====
        if os.path.exists(self._storage_file):
            try:
                self.load_from_file(self._storage_file)
                # print(f"BasinManager initialized: loaded {self.n_basins} basins from {self._storage_file}")
            except Exception as e:
                print(f"Warning: failed to load {self._storage_file}: {e}")
                print("Starting with empty basin database.")

    def save_to_file(self, filename: str = None):
        """
        Save the basin database to a pickle file.

        Persists all basin information of the current manager to disk,
        including:
        - Positions, energies, and forces of all known basins
        - Number of basins and distance threshold
        - Topology of the reference atoms object (symbols, cell, PBC)

        Parameters
        ----------
        filename : str, optional
            Path to the save file. If None, uses the storage_file
            specified at initialization (default 'basin.pkl').

        Examples
        --------
        >>> bm = BasinManager()
        >>> # ... perform some basin identification ...
        >>> bm.save_to_file("my_basins.pkl")  # save to a specific file
        >>> bm.save_to_file()  # save to the default file

        Notes
        -----
        - The file format is Python pickle; not portable across Python
          versions.
        - Saved positions are already PBC-wrapped.
        - Existing files will be overwritten.
        """
        target_file = filename if filename else self._storage_file
        
        # Build data dictionary
        data = {
            "positions": self.known_minima_positions,
            "energies": self.known_minima_energies,
            "forces": self.known_minima_forces,
            "n_basins": self.n_basins,
            "threshold": self._min_dist_threshold,
        }
        
        # Save atom system topology
        if self.atoms is not None:
            atoms_info = {
                "symbols": self.atoms.get_chemical_symbols(),
                "cell": self.atoms.cell[:],
                "pbc": self.atoms.pbc[:],
            }
            data["atoms_info"] = atoms_info
        
        try:
            with open(target_file, 'wb') as f:
                pickle.dump(data, f)
            # print(f"BasinManager: saved {self.n_basins} basins to {target_file}")
        except Exception as e:
            print(f"Error: failed to save to {target_file}: {e}")

    def load_from_file(self, filename: str = None):
        """
        Load the basin database from a pickle file.

        Restores a previously saved basin database, including all basin
        information and the reference atoms object. After loading,
        all positions are PBC-wrapped to ensure data consistency.

        Parameters
        ----------
        filename : str, optional
            Path to the load file. If None, uses the storage_file
            specified at initialization (default 'basin.pkl').

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.

        Examples
        --------
        >>> bm = BasinManager()
        >>> bm.load_from_file("my_basins.pkl")
        >>> print(f"Loaded {bm.n_basins} basins")

        Notes
        -----
        - Loading overwrites all data in the current manager.
        - If the file contains atoms_info, the reference atoms object
          is reconstructed.
        - The cache is marked as invalid after loading and will be
          rebuilt automatically at next use.
        """
        target_file = filename if filename else self._storage_file
        
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"Storage file not found: {target_file}")

        with open(target_file, 'rb') as f:
            data = pickle.load(f)
        
        # Restore basin data
        self.known_minima_positions = data.get("positions", [])
        self.known_minima_energies = data.get("energies", [])
        self.known_minima_forces = data.get("forces", [])
        self.n_basins = data.get("n_basins", 0)
        
        # Restore distance threshold
        if "threshold" in data:
            self._min_dist_threshold = data["threshold"]
        
        # Reconstruct reference atoms object
        if "atoms_info" in data:
            atoms_info = data["atoms_info"]
            symbols = atoms_info.get("symbols", [])
            cell = atoms_info.get("cell", None)
            pbc = atoms_info.get("pbc", None)
            
            if symbols:
                # Initialize with the first basin's positions (if available)
                if self.known_minima_positions:
                    positions = self.known_minima_positions[0].reshape(-1, 3)
                else:
                    positions = np.zeros((len(symbols), 3))
                
                self.atoms = Atoms(symbols=symbols, positions=positions)
                if cell is not None:
                    self.atoms.set_cell(cell)
                if pbc is not None:
                    self.atoms.set_pbc(pbc)
                
                # PBC-wrap all stored positions (for backward compatibility with old data)
                self._wrap_stored_positions()
        
        # Mark cache as invalid
        self._dirty_centers = True

    def map_atoms_to_basin(self, atoms: Atoms) -> Optional[int]:
        """
        Map an atomic structure to its corresponding basin (energy well).

        This is the core method of BasinManager. Given an atomic structure,
        it determines which known basin it belongs to via structure
        optimization and distance comparison, or registers it as a new basin.

        Algorithm flow:
        1. Check whether the input structure is close to a known basin
           (fast hit)
        2. If not close, start structure optimization
        3. During optimization, continuously check whether the structure
           enters the attraction basin of a known stable state
        4. If optimization converges to a new position, compute the
           smallest Hessian eigenvalue
        5. Eigenvalue > 0 → register as a new basin;
           eigenvalue < 0 → classify as a saddle point, return None

        Parameters
        ----------
        atoms : Atoms
            The atomic structure to classify. **Must have a calculator
            attached.** Position, cell, and PBC information are
            automatically taken into account.

        Returns
        -------
        basin_id : int or None
            - int (>= 0): ID of the successfully matched basin
              - If a known basin, returns its ID
              - If a new basin, registers it and returns the new ID
            - None: optimization converged to a saddle point (smallest
              Hessian eigenvalue < 0) or optimization did not converge

        Raises
        ------
        ValueError
            If the input atoms has no calculator attached.

        Examples
        --------
        Basic usage:

        >>> from ase.build import bulk
        >>> from ase.calculators.emt import EMT
        >>> 
        >>> bm = BasinManager(fmax=0.01, min_dist_threshold=0.8)
        >>> 
        >>> # Create structure and attach calculator
        >>> atoms = bulk('Cu', 'fcc', a=3.6)
        >>> atoms.rattle(stdev=0.1)
        >>> atoms.calc = EMT()
        >>>
        >>> # Map to a basin
        >>> basin_id = bm.map_atoms_to_basin(atoms)
        >>> 
        >>> if basin_id is not None:
        >>>     print(f"Mapped to basin {basin_id}")
        >>>     basin_atoms = bm.export_basin(basin_id)
        >>> else:
        >>>     print("Structure is a saddle point")

        Processing multiple structures:

        >>> basin_ids = []
        >>> for i, atoms in enumerate(structures):
        >>>     atoms.calc = EMT()
        >>>     bid = bm.map_atoms_to_basin(atoms)
        >>>     basin_ids.append(bid)
        >>>     print(f"Structure {i}: basin {bid}")

        Notes
        -----
        - The input atoms object is not modified; operations are performed
          on a copy.
        - The optimizer is reused for efficiency.
        - Periodic boundary conditions (PBC) are automatically accounted for.
        - Logs are written to the log file specified at initialization.
        - If verbose=False, each call outputs only the final result line;
          if verbose=True, all optimization steps are logged.
        - Saddle-point detection requires computing the smallest Hessian
          eigenvalue, which is relatively expensive.
        """
        # ===== Input validation and preparation =====
        calc = atoms.calc
        if calc is None:
            raise ValueError("Input atoms must have a calculator set.")
        
        # Create a copy to avoid modifying the original data
        atoms = atoms.copy()
        atoms.calc = calc
        
        # Save reference atoms object
        self.atoms = atoms.copy()

        # ===== Step 1: Prepare cache =====
        self._ensure_centers_cache()

        # Get current positions (PBC-wrapped)
        atoms = wrap_atoms(atoms)
        start_pos = self._get_flat_positions(atoms)
        
        # ===== Step 2: Fast-hit check =====
        if self.n_basins > 0:
            all_dists = self._calc_pbc_distances(start_pos, self._cached_centers_array)
            min_dist = np.min(all_dists)
            
            if min_dist < self._min_dist_threshold:
                # Fast hit to a known basin
                hit_id = int(np.argmin(all_dists))
                
                # Write log
                write_header = not os.path.exists(self._logfile) or os.stat(self._logfile).st_size == 0
                with open(self._logfile, 'a') as f:
                    if write_header:
                        self._write_log_header(f)
                    
                    energy = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                    self.fcalls += 1  # Count force evaluation
                    fmax = np.sqrt((forces**2).sum(axis=1).max())
                    self._write_log_line(f, 0, min_dist, "FAST", hit_id, energy, fmax, " ")
                
                # Clear references
                self._clear_calc_atoms(calc)
                atoms.calc = None
                return hit_id

        # ===== Step 3: Structure optimization =====
        opt, opt_atoms = self._get_optimizer(atoms)
        step_counter = 0
        hit_basin_id = None
        converged = False
        max_steps = 10000
        
        # Open log file
        write_header = not os.path.exists(self._logfile) or os.stat(self._logfile).st_size == 0
        
        with open(self._logfile, 'a') as f:
            if write_header:
                self._write_log_header(f)
            
            # Stepwise optimization
            while step_counter < max_steps:
                # Get current state
                opt_atoms = wrap_atoms(opt_atoms)
                pos = self._get_flat_positions(opt_atoms)
                forces = opt_atoms.get_forces()
                fmax = np.sqrt((forces**2).sum(axis=1).max())
                energy = opt_atoms.get_potential_energy()
                self.fcalls += 1  # Count force evaluation
                
                # Compute distance to the nearest basin
                min_dist = 9.9
                nearest_id = -1
                if self.n_basins > 0:
                    dists = self._calc_pbc_distances(pos, self._cached_centers_array)
                    nearest_id = int(np.argmin(dists))
                    min_dist = dists[nearest_id]
                
                # Detailed log output (verbose=True only)
                if self._verbose:
                    self._write_log_line(f, step_counter, min_dist, " ", nearest_id, energy, fmax, " ")
                
                # Check whether a known basin has been hit
                if self.n_basins > 0 and min_dist < self._min_dist_threshold:
                    hit_basin_id = nearest_id
                    break

                # Check force convergence
                if fmax < self._fmax:
                    converged = True
                    break

                # Execute one optimization step
                opt.step()
                step_counter += 1
            
            # ===== Step 4: Process optimization result =====
            
            if hit_basin_id is not None:
                # Case 1: Hit a known basin during optimization
                opt_atoms = wrap_atoms(opt_atoms)
                final_pos = self._get_flat_positions(opt_atoms)
                final_energy = opt_atoms.get_potential_energy()
                final_forces = opt_atoms.get_forces()
                self.fcalls += 1  # Count force evaluation
                final_fmax = np.sqrt((final_forces**2).sum(axis=1).max())
                
                if self.n_basins > 0:
                    dists = self._calc_pbc_distances(final_pos, self._cached_centers_array)
                    final_min_dist = dists[hit_basin_id]
                else:
                    final_min_dist = 0.0
                
                # Output final log line
                self._write_log_line(f, step_counter, final_min_dist, " ", hit_basin_id, 
                                   final_energy, final_fmax, " ")
                return hit_basin_id
            
            if converged:
                # Case 2: Optimization converged to a new position
                opt_atoms = wrap_atoms(opt_atoms)
                final_pos = self._get_flat_positions(opt_atoms)
                final_energy = opt_atoms.get_potential_energy()
                final_forces = opt_atoms.get_forces()
                self.fcalls += 1  # Count force evaluation
                final_fmax = np.sqrt((final_forces**2).sum(axis=1).max())
                
                # Re-check proximity to known basins (last step may converge to a known basin)
                is_new = True
                final_id = -1
                min_dist_final = 0.0
                
                if self.n_basins > 0:
                    dists = self._calc_pbc_distances(final_pos, self._cached_centers_array)
                    nearest_id = int(np.argmin(dists))
                    min_dist_final = dists[nearest_id]
                    
                    if min_dist_final < self._min_dist_threshold:
                        # Converged to a known basin
                        final_id = nearest_id
                        is_new = False
                
                if is_new:
                    # Case 2a: Genuinely new position — determine basin or saddle point
                    # Compute smallest Hessian eigenvalue
                    eig_val = self._calc_min_eigenvalue(opt_atoms)
                    
                    if eig_val < 0:
                        # Saddle point — do not register
                        self._write_log_line(f, step_counter, 0.0, "SADDLE", -1, 
                                           final_energy, final_fmax, eig_val)
                        return None
                    
                    # New basin — register
                    final_id = self.n_basins
                    self.n_basins += 1
                    self.known_minima_positions.append(final_pos)
                    self.known_minima_energies.append(final_energy)
                    self.known_minima_forces.append(final_forces)
                    self._dirty_centers = True
                    
                    # Compute minimum distance to other basins (for logging)
                    if self.n_basins > 1:
                        other_dists = self._calc_pbc_distances(final_pos, 
                                                              self._cached_centers_array)
                        min_dist_log = np.min(other_dists) if len(other_dists) > 0 else 0.0
                    else:
                        min_dist_log = 0.0
                    
                    self._write_log_line(f, step_counter, min_dist_log, " ", final_id, 
                                       final_energy, final_fmax, eig_val)
                    return final_id
                else:
                    # Case 2b: Converged to a known basin
                    self._write_log_line(f, step_counter, min_dist_final, " ", final_id, 
                                       final_energy, final_fmax, " ")
                    return final_id
            
            # Case 3: Reached max steps without convergence
            f.write(f"# WARNING: Optimization did not converge within {max_steps} steps\n")
            return None

    def export_basin(self, basin_id: int) -> Atoms:
        """
        Export the atomic structure of a specified basin.

        Returns an Atoms object with full information for the specified
        basin, including positions, energy, and forces. Positions are
        automatically PBC-wrapped to lie within the primary cell.

        Parameters
        ----------
        basin_id : int
            ID of the basin to export. Must be in the valid range
            [0, n_basins-1].

        Returns
        -------
        atoms : Atoms
            The basin's atomic structure, containing:
            - Optimized atomic positions (wrapped to the primary cell)
            - Energy and forces set via a SinglePointCalculator
            - System topology (symbols, cell, PBC)

        Raises
        ------
        ValueError
            If basin_id is out of range, or the atoms object has not
            been initialized.

        Examples
        --------
        >>> bm = BasinManager()
        >>> # ... identify some basins ...
        >>>
        >>> # Export all basins
        >>> for i in range(bm.n_basins):
        >>>     atoms = bm.export_basin(i)
        >>>     print(f"Basin {i}: E = {atoms.get_potential_energy():.4f} eV")
        >>>     atoms.write(f"basin_{i}.xyz")
        >>>
        >>> # Export the lowest-energy basin
        >>> lowest_id = np.argmin(bm.known_minima_energies)
        >>> lowest_atoms = bm.export_basin(lowest_id)

        Notes
        -----
        - The returned Atoms object is an independent copy; modifications
          do not affect BasinManager internal data.
        - Positions are PBC-wrapped.
        - Energy and forces are set via a SinglePointCalculator so no
          recalculation is needed.
        """
        # Validate basin_id
        if basin_id < 0 or basin_id >= self.n_basins:
            raise ValueError(
                f"Invalid basin_id {basin_id}. "
                f"Valid range: [0, {self.n_basins - 1}]"
            )
        
        if self.atoms is None:
            raise ValueError(
                "Reference atoms object not initialized. "
                "This should not happen if BasinManager was used correctly."
            )
        
        # Create atoms object
        atoms = self.atoms.copy()
        atoms.set_positions(self.known_minima_positions[basin_id].reshape(-1, 3))
        atoms = wrap_atoms(atoms)
        
        # Attach SinglePointCalculator
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=self.known_minima_energies[basin_id],
            forces=self.known_minima_forces[basin_id],
        )
        
        return atoms

    # ========================================
    # Internal helper methods (users normally do not call these directly)
    # ========================================

    def _get_flat_positions(self, atoms: Atoms) -> np.ndarray:
        """
        Get flattened atomic position array.

        Parameters
        ----------
        atoms : Atoms
            Atoms object

        Returns
        -------
        positions : np.ndarray, shape (N_atoms * 3,)
            Flattened position array
        """
        return atoms.get_positions().flatten()

    def _wrap_stored_positions(self):
        """
        PBC-wrap all stored basin positions.

        Used for backward-compatibility when loading old data,
        ensuring all positions lie within the primary cell.
        """
        if not self.known_minima_positions or self.atoms is None:
            return
        
        wrapped_positions = []
        for pos in self.known_minima_positions:
            temp_atoms = self.atoms.copy()
            temp_atoms.set_positions(pos.reshape(-1, 3))
            temp_atoms = wrap_atoms(temp_atoms)
            wrapped_positions.append(temp_atoms.get_positions().flatten())
        
        self.known_minima_positions = wrapped_positions

    def _calc_pbc_distances(self, pos: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Compute PBC-aware distances using the minimum-image convention.

        Computes the distance from a given position to all basin centers
        under periodic boundary conditions.

        Parameters
        ----------
        pos : np.ndarray, shape (N_atoms * 3,)
            Current position (flattened)
        centers : np.ndarray, shape (N_basins, N_atoms * 3)
            Position array of all basin centers

        Returns
        -------
        distances : np.ndarray, shape (N_basins,)
            L2 distance to each basin center
        """
        if self.atoms is None:
            return np.linalg.norm(centers - pos, axis=1)
        
        pbc = self.atoms.get_pbc()
        if not np.any(pbc):
            return np.linalg.norm(centers - pos, axis=1)
        
        cell = self.atoms.get_cell()
        if np.abs(np.linalg.det(cell)) < 1e-10:
            return np.linalg.norm(centers - pos, axis=1)
        
        n_atoms = len(pos) // 3
        n_basins = len(centers)
        
        pos_3d = pos.reshape(-1, 3)  # (n_atoms, 3)
        centers_3d = centers.reshape(n_basins, n_atoms, 3)  # (n_basins, n_atoms, 3)
        
        # Compute displacement vectors
        diff = centers_3d - pos_3d  # (n_basins, n_atoms, 3)
        
        # Convert to fractional coordinates
        cell_inv = np.linalg.inv(cell)
        scaled_diff = diff @ cell_inv  # (n_basins, n_atoms, 3)
        
        # Apply minimum-image convention
        for i in range(3):
            if pbc[i]:
                scaled_diff[:, :, i] = scaled_diff[:, :, i] - np.round(scaled_diff[:, :, i])
        
        # Convert back to Cartesian coordinates
        min_diff = scaled_diff @ cell  # (n_basins, n_atoms, 3)
        
        # Compute L2 distance
        distances = np.sqrt(np.sum(min_diff ** 2, axis=(1, 2)))  # (n_basins,)
        
        return distances

    def _ensure_centers_cache(self):
        """
        Maintain the basin-centers cache.

        Rebuilds the NumPy array only when new basins have been added
        (_dirty_centers=True) to avoid repeated list-to-array conversion
        overhead.
        """
        if self._dirty_centers:
            if self.known_minima_positions:
                self._cached_centers_array = np.array(self.known_minima_positions)
            else:
                self._cached_centers_array = np.empty((0, 0))
            self._dirty_centers = False

    def _get_mmcalc(self, atoms: Atoms) -> MinModeCalculator:
        """
        Get or create a MinModeCalculator instance.

        Used to compute the smallest Hessian eigenvalue for determining
        basin vs. saddle point.

        Parameters
        ----------
        atoms : Atoms
            Atoms object (must have a calculator attached)

        Returns
        -------
        mmcalc : MinModeCalculator
            Minimum-mode calculator instance
        """
        if self._mm_calc is None:
            self._mm_calc = MinModeCalculator(std_calc=atoms.calc)
        else:
            self._mm_calc.std_calc = atoms.calc
        return self._mm_calc

    def _calc_min_eigenvalue(self, atoms: Atoms) -> float:
        """
        Compute the smallest eigenvalue of the Hessian matrix.

        Used to determine whether an optimization-converged point is a
        basin (eigenvalue > 0) or a saddle point (eigenvalue < 0).
        This is relatively expensive and uses finite differences to
        compute the Hessian.

        Parameters
        ----------
        atoms : Atoms
            Atomic structure after optimization convergence

        Returns
        -------
        eig_val : float
            Smallest eigenvalue of the Hessian matrix

        Notes
        -----
        Before each call, min_mode is forcibly reset to None so that
        the Lanczos iteration starts from a random vector. This prevents
        reusing the positive-curvature direction found at a previous
        basin bottom as the initial guess, which would confine the
        Krylov subspace to an orthogonal complement, miss a true
        imaginary-frequency direction, and misclassify a saddle point
        as a basin. After the calculation, min_mode is also cleared to
        prevent cross-call residue contamination.

        Note: BasinManager._mm_calc is used only within this method.
        Clearing it here does not affect HybridBoostCalculator._mm_calc
        or the warm-start logic of other modules.
        """
        mm_atoms = atoms.copy()
        mm_calc = self._get_mmcalc(atoms)
        # Force restart from a random vector to prevent inheriting a
        # positive-curvature min_mode from a previous basin bottom as
        # the initial guess, which would cause Lanczos to miss the
        # true negative-curvature direction at a saddle and return a
        # false-positive eigenvalue.
        mm_calc.parameters["min_mode"] = None
        mm_atoms.calc = mm_calc
        mm_atoms.calc.calculate(mm_atoms)
        eig_val = mm_atoms.calc.parameters["eig_values"][0]
        
        # Accumulate MinModeCalculator's fcalls
        self.fcalls += mm_calc.fcalls
        mm_calc.fcalls = 0
        
        # Clean up temporary references and clear min_mode again to
        # prevent residue from contaminating the next call
        mm_atoms.calc = None
        if hasattr(mm_calc, 'atoms'):
            mm_calc.atoms = None
        if hasattr(mm_calc, 'results'):
            mm_calc.results = {}
        mm_calc.parameters["min_mode"] = None
        
        return eig_val

    def _clear_calc_atoms(self, calc):
        """
        Clear the internal atoms reference of a calculator.

        Prevents memory leaks caused by circular references.

        Parameters
        ----------
        calc : Calculator
            The calculator to clean up.
        """
        if calc is None:
            return
        if hasattr(calc, 'atoms'):
            calc.atoms = None
        if hasattr(calc, 'results'):
            calc.results = {}

    def _get_optimizer(self, atoms: Atoms):
        """
        Get or reuse an optimizer instance.

        Creates the optimizer on the first call; on subsequent calls
        reuses it and resets its internal state for efficiency.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to optimize

        Returns
        -------
        optimizer : Optimizer
            Optimizer instance
        opt_atoms : Atoms
            Atoms object bound to the optimizer
        """
        self._opt_atoms = atoms.copy()
        self._opt_atoms.calc = atoms.calc
        self._optimizer = self._optimizer_class(self._opt_atoms, logfile=None)
        
        return self._optimizer, self._opt_atoms

    def _write_log_header(self, file_handle):
        """
        Write the log-file header.

        Parameters
        ----------
        file_handle : file object
            Open file handle
        """
        header = (
            f"{'Step':>6} "
            f"{'Min-D':>6} "
            f"{'Reason':>6} "
            f"{'Basin':>6} "
            f"{'Energy':>12} "
            f"{'fmax':>12} "
            f"{'EigVal':>12}\n"
        )
        file_handle.write(header)

    def _write_log_line(self, file_handle, step, min_dist, reason, basin_id, 
                       energy, fmax, eig_val):
        """
        Format and write a single log line.

        Parameters
        ----------
        file_handle : file object
            Open file handle
        step : int
            Current optimization step
        min_dist : float
            Distance to the nearest basin
        reason : str or float
            Termination reason ("FAST" = fast hit, "SADDLE" = saddle
            point, " " = normal)
        basin_id : int
            Basin ID (–1 indicates a saddle point)
        energy : float
            Current energy
        fmax : float
            Current maximum force component
        eig_val : float or str
            Smallest Hessian eigenvalue (" " means not computed)
        """
        # Format reason
        if isinstance(reason, str):
            reason_str = f"{reason:>6s}"
        else:
            reason_str = f"{reason:>6.2f}"
        
        # Format eig_val
        if isinstance(eig_val, str):
            eig_str = f"{eig_val:>12s}"
        else:
            eig_str = f"{eig_val:>12.4f}"
        
        line = (
            f"{step:>6d} "
            f"{min_dist:>6.2f} "
            f"{reason_str} "
            f"{basin_id:>6d} "
            f"{energy:>12.6f} "
            f"{fmax:>12.4f} "
            f"{eig_str}\n"
        )
        file_handle.write(line)