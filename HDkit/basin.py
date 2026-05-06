#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: Qian Lixiang
# @Email: 649811459@qq.com
__author__ = "Qian Lixiang"
__email__ = "649811459@qq.com"

"""
势能面稳态(Basin)管理模块

本模块提供 BasinManager 类，用于高效识别、存储和管理分子动力学模拟中的势能面稳态。
主要功能包括：
- 通过结构优化将原子构型映射到对应的稳态（basin）
- 智能缓存已知稳态，快速识别重复访问的结构
- 考虑周期性边界条件的距离计算
- 自动判断鞍点（saddle point）并拒绝注册
- 持久化存储/加载稳态数据库

典型用法:
    >>> from ase import Atoms
    >>> from ase.calculators.emt import EMT
    >>> 
    >>> # 创建管理器
    >>> bm = BasinManager(min_dist_threshold=1.0, fmax=0.01)
    >>>
    >>> # 识别结构对应的稳态ID
    >>> atoms = Atoms('Cu4', ...)
    >>> atoms.calc = EMT()
    >>> basin_id = bm.map_atoms_to_basin(atoms)
    >>>
    >>> # 导出已知稳态
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
    对原子位置(positions)进行周期性边界条件(PBC)包裹。

    Parameters
    ----------
    atoms : Atoms
        需要包裹的原子对象。

    Returns
    -------
    Atoms
        包裹后的原子对象。
    """
    pbc = atoms.get_pbc()
    if not any(pbc):
        return atoms
    cell = atoms.get_cell()
    if np.abs(np.linalg.det(cell)) < 1e-10:
        return atoms
    pos = atoms.get_positions()

    # 转换为分数坐标
    cell_inv = np.linalg.inv(cell)
    scaled = pos @ cell_inv

    # 对有周期性的方向进行包裹
    for i in range(3):
        if pbc[i]:
            scaled[:, i] = scaled[:, i] % 1.0

    # 转换回笛卡尔坐标
    pos_wrapped = scaled @ cell

    atoms.set_positions(pos_wrapped)
    return atoms


class BasinFoundException(Exception):
    """
    用于在优化过程中捕获稳态命中的异常信号。
    
    当优化器在迭代过程中发现原子结构进入已知稳态的吸引域时抛出。
    
    Attributes:
        basin_id: 命中的稳态ID
    """
    def __init__(self, basin_id: int):
        self.basin_id = basin_id
        super().__init__(f"Basin {basin_id} found")


class BasinManager:
    """
    势能面稳态(Basin)管理器
    
    本类通过结构优化将原子构型映射到势能面的局部最小值（稳态），并维护已知稳态的数据库。
    利用智能缓存和周期性边界条件优化，实现高效的稳态识别。
    
    工作流程：
    1. 输入原子结构（必须已设置计算器）
    2. 检查是否接近已知稳态（通过距离阈值判断）
    3. 如果不接近，进行结构优化
    4. 优化过程中持续检查是否进入已知稳态吸引域
    5. 如果优化收敛到新位置，计算Hessian最小特征值判断是否为稳态
    6. 如果特征值>0，注册为新稳态；如果<0，判定为鞍点并拒绝注册
    
    Parameters
    ----------
    min_dist_threshold : float, default=1.0
        判定两个结构是否属于同一稳态的距离阈值（单位：Å）。
        当两个结构的原子位置差异（L2范数）小于此值时，认为它们属于同一稳态。
        建议根据系统大小和灵敏度需求调整：
        - 较小系统（<100原子）：0.5-1.0 Å
        - 较大系统（>100原子）：1.0-2.0 Å
    
    logfile : str, default='rlx.log'
        优化过程日志文件路径。记录每次map_atoms_to_basin调用的详细信息：
        - Step: 优化步数
        - Min-D: 到最近稳态的距离
        - Reason: 终止原因（FAST=快速命中，SADDLE=鞍点，空=正常收敛）
        - Basin: 命中的稳态ID（-1表示鞍点）
        - Energy: 当前势能
        - fmax: 当前最大力
        - EigVal: Hessian最小特征值（仅新稳态时计算）
    
    fmax : float, default=1e-2
        结构优化的力收敛标准（单位：eV/Å）。
        当所有原子的最大力分量小于此值时，认为优化收敛。
    
    optimizer_class : Type[Optimizer], optional
        使用的ASE优化器类。默认使用BFGSLineSearch。
        可选：FIRE, LBFGS, GPMin等ASE支持的优化器。
    
    storage_file : str, default='basin.pkl'
        持久化存储文件路径。保存/加载已知稳态数据库，包括：
        - 稳态位置（考虑周期性边界条件）
        - 稳态能量和力
        - 原子系统信息（符号、晶胞、周期性边界条件）
        如果文件存在，初始化时会自动加载。
    
    verbose : bool, default=False
        是否输出详细日志。
        - True: 每个优化步骤都输出到logfile
        - False: 仅输出最终结果
    
    Attributes
    ----------
    n_basins : int
        当前已注册的稳态数量
    
    known_minima_positions : List[np.ndarray]
        已知稳态的原子位置列表，每个元素形状为 (N_atoms*3,)
    
    known_minima_energies : List[float]
        已知稳态的能量列表
    
    known_minima_forces : List[np.ndarray]
        已知稳态的力列表，每个元素形状为 (N_atoms, 3)
    
    atoms : Atoms
        参考原子对象，保存系统的拓扑信息（符号、晶胞、PBC）
    
    Examples
    --------
    基本用法：
    
    >>> from ase.build import bulk
    >>> from ase.calculators.emt import EMT
    >>> 
    >>> # 创建管理器
    >>> bm = BasinManager(min_dist_threshold=0.8, fmax=0.01, verbose=True)
    >>>
    >>> # 创建原子结构并设置计算器
    >>> atoms = bulk('Cu', 'fcc', a=3.6).repeat((2, 2, 2))
    >>> atoms.rattle(stdev=0.1)  # 随机扰动
    >>> atoms.calc = EMT()
    >>>
    >>> # 识别稳态
    >>> basin_id = bm.map_atoms_to_basin(atoms)
    >>> print(f"Structure mapped to basin {basin_id}")
    >>>
    >>> # 导出稳态结构
    >>> if basin_id is not None:
    >>>     basin_atoms = bm.export_basin(basin_id)
    >>>     print(f"Basin energy: {basin_atoms.get_potential_energy():.4f} eV")
    >>>
    >>> # 保存数据库以供后续使用
    >>> bm.save_to_file("my_basins.pkl")
    
    从已有数据库加载：
    
    >>> # 下次运行时自动加载
    >>> bm = BasinManager(storage_file="my_basins.pkl")
    >>> print(f"Loaded {bm.n_basins} known basins")
    
    Notes
    -----
    - 本类自动处理周期性边界条件（PBC），使用最小镜像约定计算距离
    - 优化器实例会被复用以提高效率，避免重复初始化开销
    - 对于大规模系统，建议适当增大min_dist_threshold和fmax以加快计算
    - 鞍点判断依赖于Hessian最小特征值计算，这是一个相对昂贵的操作，
      仅在优化收敛到新位置时执行
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

        # ===== 配置参数 =====
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

        # ===== 内部工具 =====
        self._mm_calc: MinModeCalculator = None  # 用于计算Hessian最小特征值
        self._opt_atoms: Optional[Atoms] = None  # 优化器绑定的atoms对象
        self._optimizer: Optional[Optimizer] = None  # 复用的优化器实例
        self.fcalls: int = 0  # 力计算次数统计

        # ===== 稳态数据库 =====
        self.atoms: Optional[Atoms] = None  # 参考原子对象（不含calc）
        self.known_minima_positions: List[np.ndarray] = []  # 稳态位置列表
        self.known_minima_energies: List[float] = []  # 稳态能量列表
        self.known_minima_forces: List[np.ndarray] = []  # 稳态力列表
        self.n_basins = 0  # 已注册稳态数量

        # ===== 性能优化缓存 =====
        self._dirty_centers = True  # 缓存失效标记
        self._cached_centers_array: Optional[np.ndarray] = None  # 稳态位置缓存(Nb, 3N)

        # ===== 从文件加载已有数据（如果存在）=====
        if os.path.exists(self._storage_file):
            try:
                self.load_from_file(self._storage_file)
                # print(f"BasinManager initialized: loaded {self.n_basins} basins from {self._storage_file}")
            except Exception as e:
                print(f"Warning: failed to load {self._storage_file}: {e}")
                print("Starting with empty basin database.")

    def save_to_file(self, filename: str = None):
        """
        保存稳态数据库到pickle文件
        
        将当前管理器的所有稳态信息持久化到磁盘，包括：
        - 所有已知稳态的位置、能量、力
        - 稳态数量和距离阈值
        - 参考原子对象的拓扑信息（符号、晶胞、PBC）
        
        Parameters
        ----------
        filename : str, optional
            保存文件路径。如果为None，使用初始化时指定的storage_file（默认'basin.pkl'）
        
        Examples
        --------
        >>> bm = BasinManager()
        >>> # ... 进行一些稳态识别 ...
        >>> bm.save_to_file("my_basins.pkl")  # 保存到指定文件
        >>> bm.save_to_file()  # 保存到默认文件
        
        Notes
        -----
        - 文件格式为Python pickle，不可跨Python版本移植
        - 保存的位置已考虑周期性边界条件进行包裹
        - 如果文件已存在，会被覆盖
        """
        target_file = filename if filename else self._storage_file
        
        # 构建数据字典
        data = {
            "positions": self.known_minima_positions,
            "energies": self.known_minima_energies,
            "forces": self.known_minima_forces,
            "n_basins": self.n_basins,
            "threshold": self._min_dist_threshold,
        }
        
        # 保存原子系统拓扑信息
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
        从pickle文件加载稳态数据库
        
        恢复之前保存的稳态数据库，包括所有稳态信息和参考原子对象。
        加载后会自动对所有位置进行周期性包裹，确保数据一致性。
        
        Parameters
        ----------
        filename : str, optional
            加载文件路径。如果为None，使用初始化时指定的storage_file（默认'basin.pkl'）
        
        Raises
        ------
        FileNotFoundError
            如果指定文件不存在
        
        Examples
        --------
        >>> bm = BasinManager()
        >>> bm.load_from_file("my_basins.pkl")
        >>> print(f"Loaded {bm.n_basins} basins")
        
        Notes
        -----
        - 加载会覆盖当前管理器的所有数据
        - 如果文件中包含atoms_info，会重建参考原子对象
        - 加载后缓存会被标记为失效，下次使用时自动重建
        """
        target_file = filename if filename else self._storage_file
        
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"Storage file not found: {target_file}")

        with open(target_file, 'rb') as f:
            data = pickle.load(f)
        
        # 恢复稳态数据
        self.known_minima_positions = data.get("positions", [])
        self.known_minima_energies = data.get("energies", [])
        self.known_minima_forces = data.get("forces", [])
        self.n_basins = data.get("n_basins", 0)
        
        # 恢复距离阈值
        if "threshold" in data:
            self._min_dist_threshold = data["threshold"]
        
        # 重建参考原子对象
        if "atoms_info" in data:
            atoms_info = data["atoms_info"]
            symbols = atoms_info.get("symbols", [])
            cell = atoms_info.get("cell", None)
            pbc = atoms_info.get("pbc", None)
            
            if symbols:
                # 使用第一个稳态的位置初始化（如果有）
                if self.known_minima_positions:
                    positions = self.known_minima_positions[0].reshape(-1, 3)
                else:
                    positions = np.zeros((len(symbols), 3))
                
                self.atoms = Atoms(symbols=symbols, positions=positions)
                if cell is not None:
                    self.atoms.set_cell(cell)
                if pbc is not None:
                    self.atoms.set_pbc(pbc)
                
                # 对所有已存储的位置进行周期性包裹（兼容旧数据）
                self._wrap_stored_positions()
        
        # 标记缓存失效
        self._dirty_centers = True

    def map_atoms_to_basin(self, atoms: Atoms) -> Optional[int]:
        """
        将原子结构映射到对应的稳态(Basin)
        
        这是BasinManager的核心方法。给定一个原子结构，通过结构优化和距离比较，
        判断它属于哪个已知稳态，或者注册为新的稳态。
        
        算法流程：
        1. 检查输入结构是否接近已知稳态（快速命中）
        2. 如果不接近，启动结构优化
        3. 优化过程中持续检查是否进入已知稳态吸引域
        4. 如果优化收敛到新位置，计算Hessian最小特征值
        5. 特征值>0 → 注册新稳态；特征值<0 → 判定为鞍点，返回None
        
        Parameters
        ----------
        atoms : Atoms
            待分类的原子结构，**必须已设置calculator**。
            结构的位置、晶胞和PBC信息会被自动考虑。
        
        Returns
        -------
        basin_id : int or None
            - int (>= 0): 成功映射到的稳态ID
              - 如果是已知稳态，返回其ID
              - 如果是新稳态，注册后返回新ID
            - None: 优化收敛到鞍点（Hessian最小特征值<0）或优化未收敛
        
        Raises
        ------
        ValueError
            如果输入的atoms未设置calculator
        
        Examples
        --------
        基本用法：
        
        >>> from ase.build import bulk
        >>> from ase.calculators.emt import EMT
        >>> 
        >>> bm = BasinManager(fmax=0.01, min_dist_threshold=0.8)
        >>> 
        >>> # 创建结构并设置计算器
        >>> atoms = bulk('Cu', 'fcc', a=3.6)
        >>> atoms.rattle(stdev=0.1)
        >>> atoms.calc = EMT()
        >>>
        >>> # 映射到稳态
        >>> basin_id = bm.map_atoms_to_basin(atoms)
        >>> 
        >>> if basin_id is not None:
        >>>     print(f"Mapped to basin {basin_id}")
        >>>     basin_atoms = bm.export_basin(basin_id)
        >>> else:
        >>>     print("Structure is a saddle point")
        
        处理多个结构：
        
        >>> basin_ids = []
        >>> for i, atoms in enumerate(structures):
        >>>     atoms.calc = EMT()
        >>>     bid = bm.map_atoms_to_basin(atoms)
        >>>     basin_ids.append(bid)
        >>>     print(f"Structure {i}: basin {bid}")
        
        Notes
        -----
        - 输入的atoms对象不会被修改，内部使用副本进行操作
        - 优化器会被复用以提高效率
        - 周期性边界条件（PBC）会被自动考虑
        - 日志会写入初始化时指定的logfile
        - 如果verbose=False，每次调用仅输出最终结果行；
          如果verbose=True，输出所有优化步骤
        - 判断鞍点需要计算Hessian最小特征值，这是一个相对昂贵的操作
        """
        # ===== 输入验证和准备 =====
        calc = atoms.calc
        if calc is None:
            raise ValueError("Input atoms must have a calculator set.")
        
        # 创建副本避免修改原始数据
        atoms = atoms.copy()
        atoms.calc = calc
        
        # 保存参考原子对象
        self.atoms = atoms.copy()

        # ===== 步骤1: 准备缓存 =====
        self._ensure_centers_cache()

        # 获取当前位置（考虑周期性包裹）
        atoms = wrap_atoms(atoms)
        start_pos = self._get_flat_positions(atoms)
        
        # ===== 步骤2: 快速命中检查 =====
        if self.n_basins > 0:
            all_dists = self._calc_pbc_distances(start_pos, self._cached_centers_array)
            min_dist = np.min(all_dists)
            
            if min_dist < self._min_dist_threshold:
                # 快速命中已知稳态
                hit_id = int(np.argmin(all_dists))
                
                # 记录日志
                write_header = not os.path.exists(self._logfile) or os.stat(self._logfile).st_size == 0
                with open(self._logfile, 'a') as f:
                    if write_header:
                        self._write_log_header(f)
                    
                    energy = atoms.get_potential_energy()
                    forces = atoms.get_forces()
                    self.fcalls += 1  # 统计力计算次数
                    fmax = np.sqrt((forces**2).sum(axis=1).max())
                    self._write_log_line(f, 0, min_dist, "FAST", hit_id, energy, fmax, " ")
                
                # 清理引用
                self._clear_calc_atoms(calc)
                atoms.calc = None
                return hit_id

        # ===== 步骤3: 结构优化 =====
        opt, opt_atoms = self._get_optimizer(atoms)
        step_counter = 0
        hit_basin_id = None
        converged = False
        max_steps = 10000
        
        # 打开日志文件
        write_header = not os.path.exists(self._logfile) or os.stat(self._logfile).st_size == 0
        
        with open(self._logfile, 'a') as f:
            if write_header:
                self._write_log_header(f)
            
            # 逐步优化
            while step_counter < max_steps:
                # 获取当前状态
                opt_atoms = wrap_atoms(opt_atoms)
                pos = self._get_flat_positions(opt_atoms)
                forces = opt_atoms.get_forces()
                fmax = np.sqrt((forces**2).sum(axis=1).max())
                energy = opt_atoms.get_potential_energy()
                self.fcalls += 1  # 统计力计算次数
                
                # 计算到最近稳态的距离
                min_dist = 9.9
                nearest_id = -1
                if self.n_basins > 0:
                    dists = self._calc_pbc_distances(pos, self._cached_centers_array)
                    nearest_id = int(np.argmin(dists))
                    min_dist = dists[nearest_id]
                
                # 详细日志输出（仅verbose=True）
                if self._verbose:
                    self._write_log_line(f, step_counter, min_dist, " ", nearest_id, energy, fmax, " ")
                
                # 检查是否命中已知稳态
                if self.n_basins > 0 and min_dist < self._min_dist_threshold:
                    hit_basin_id = nearest_id
                    break

                # 检查是否力收敛
                if fmax < self._fmax:
                    converged = True
                    break

                # 执行一步优化
                opt.step()
                step_counter += 1
            
            # ===== 步骤4: 处理优化结果 =====
            
            if hit_basin_id is not None:
                # 情况1: 优化过程中命中已知稳态
                opt_atoms = wrap_atoms(opt_atoms)
                final_pos = self._get_flat_positions(opt_atoms)
                final_energy = opt_atoms.get_potential_energy()
                final_forces = opt_atoms.get_forces()
                self.fcalls += 1  # 统计力计算次数
                final_fmax = np.sqrt((final_forces**2).sum(axis=1).max())
                
                if self.n_basins > 0:
                    dists = self._calc_pbc_distances(final_pos, self._cached_centers_array)
                    final_min_dist = dists[hit_basin_id]
                else:
                    final_min_dist = 0.0
                
                # 输出最终日志行
                self._write_log_line(f, step_counter, final_min_dist, " ", hit_basin_id, 
                                   final_energy, final_fmax, " ")
                return hit_basin_id
            
            if converged:
                # 情况2: 优化收敛到新位置
                opt_atoms = wrap_atoms(opt_atoms)
                final_pos = self._get_flat_positions(opt_atoms)
                final_energy = opt_atoms.get_potential_energy()
                final_forces = opt_atoms.get_forces()
                self.fcalls += 1  # 统计力计算次数
                final_fmax = np.sqrt((final_forces**2).sum(axis=1).max())
                
                # 再次检查是否接近已知稳态（可能在最后一步收敛到已知稳态）
                is_new = True
                final_id = -1
                min_dist_final = 0.0
                
                if self.n_basins > 0:
                    dists = self._calc_pbc_distances(final_pos, self._cached_centers_array)
                    nearest_id = int(np.argmin(dists))
                    min_dist_final = dists[nearest_id]
                    
                    if min_dist_final < self._min_dist_threshold:
                        # 收敛到已知稳态
                        final_id = nearest_id
                        is_new = False
                
                if is_new:
                    # 情况2a: 真正的新位置，需要判断是稳态还是鞍点
                    # 计算Hessian最小特征值
                    eig_val = self._calc_min_eigenvalue(opt_atoms)
                    
                    if eig_val < 0:
                        # 鞍点，不注册
                        self._write_log_line(f, step_counter, 0.0, "SADDLE", -1, 
                                           final_energy, final_fmax, eig_val)
                        return None
                    
                    # 新稳态，注册
                    final_id = self.n_basins
                    self.n_basins += 1
                    self.known_minima_positions.append(final_pos)
                    self.known_minima_energies.append(final_energy)
                    self.known_minima_forces.append(final_forces)
                    self._dirty_centers = True
                    
                    # 计算到其他稳态的最小距离（用于日志）
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
                    # 情况2b: 收敛到已知稳态
                    self._write_log_line(f, step_counter, min_dist_final, " ", final_id, 
                                       final_energy, final_fmax, " ")
                    return final_id
            
            # 情况3: 达到最大步数未收敛
            f.write(f"# WARNING: Optimization did not converge within {max_steps} steps\n")
            return None

    def export_basin(self, basin_id: int) -> Atoms:
        """
        导出指定稳态的原子结构
        
        返回一个包含指定稳态完整信息的Atoms对象，包括位置、能量、力。
        位置已自动进行周期性包裹，确保在主晶胞内。
        
        Parameters
        ----------
        basin_id : int
            要导出的稳态ID，必须在有效范围内 [0, n_basins-1]
        
        Returns
        -------
        atoms : Atoms
            稳态的原子结构，包含：
            - 优化后的原子位置（已包裹到主晶胞）
            - 通过SinglePointCalculator设置的能量和力
            - 原子系统的拓扑信息（符号、晶胞、PBC）
        
        Raises
        ------
        ValueError
            如果basin_id超出有效范围，或atoms对象未初始化
        
        Examples
        --------
        >>> bm = BasinManager()
        >>> # ... 识别一些稳态 ...
        >>>
        >>> # 导出所有稳态
        >>> for i in range(bm.n_basins):
        >>>     atoms = bm.export_basin(i)
        >>>     print(f"Basin {i}: E = {atoms.get_potential_energy():.4f} eV")
        >>>     atoms.write(f"basin_{i}.xyz")
        >>>
        >>> # 导出能量最低的稳态
        >>> lowest_id = np.argmin(bm.known_minima_energies)
        >>> lowest_atoms = bm.export_basin(lowest_id)
        
        Notes
        -----
        - 返回的Atoms对象是独立副本，修改不会影响BasinManager内部数据
        - 位置已考虑周期性边界条件进行包裹
        - 能量和力通过SinglePointCalculator设置，无需重新计算
        """
        # 验证basin_id
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
        
        # 创建原子对象
        atoms = self.atoms.copy()
        atoms.set_positions(self.known_minima_positions[basin_id].reshape(-1, 3))
        atoms = wrap_atoms(atoms)
        
        # 设置SinglePointCalculator
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=self.known_minima_energies[basin_id],
            forces=self.known_minima_forces[basin_id],
        )
        
        return atoms

    # ========================================
    # 内部辅助方法（用户通常不需要直接调用）
    # ========================================

    def _get_flat_positions(self, atoms: Atoms) -> np.ndarray:
        """
        获取扁平化的原子位置数组
        
        Parameters
        ----------
        atoms : Atoms
            原子对象
        
        Returns
        -------
        positions : np.ndarray, shape (N_atoms * 3,)
            扁平化的位置数组
        """
        return atoms.get_positions().flatten()

    def _wrap_stored_positions(self):
        """
        对所有已存储的稳态位置进行周期性包裹
        
        用于加载旧数据时的兼容性处理，确保所有位置都在主晶胞内。
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
        计算考虑周期性边界条件的距离
        
        使用最小镜像约定计算给定位置到所有稳态中心的距离。
        
        Parameters
        ----------
        pos : np.ndarray, shape (N_atoms * 3,)
            当前位置（扁平化）
        centers : np.ndarray, shape (N_basins, N_atoms * 3)
            所有稳态中心的位置数组
        
        Returns
        -------
        distances : np.ndarray, shape (N_basins,)
            到每个稳态中心的L2距离
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
        
        # 计算位移向量
        diff = centers_3d - pos_3d  # (n_basins, n_atoms, 3)
        
        # 转换为分数坐标
        cell_inv = np.linalg.inv(cell)
        scaled_diff = diff @ cell_inv  # (n_basins, n_atoms, 3)
        
        # 应用最小镜像约定
        for i in range(3):
            if pbc[i]:
                scaled_diff[:, :, i] = scaled_diff[:, :, i] - np.round(scaled_diff[:, :, i])
        
        # 转换回笛卡尔坐标
        min_diff = scaled_diff @ cell  # (n_basins, n_atoms, 3)
        
        # 计算L2距离
        distances = np.sqrt(np.sum(min_diff ** 2, axis=(1, 2)))  # (n_basins,)
        
        return distances

    def _ensure_centers_cache(self):
        """
        维护稳态中心缓存
        
        仅当有新稳态加入时(_dirty_centers=True)才重新构建numpy数组，
        避免重复的列表到数组转换开销。
        """
        if self._dirty_centers:
            if self.known_minima_positions:
                self._cached_centers_array = np.array(self.known_minima_positions)
            else:
                self._cached_centers_array = np.empty((0, 0))
            self._dirty_centers = False

    def _get_mmcalc(self, atoms: Atoms) -> MinModeCalculator:
        """
        获取或创建MinModeCalculator实例
        
        用于计算Hessian最小特征值以判断稳态/鞍点。
        
        Parameters
        ----------
        atoms : Atoms
            原子对象（需要已设置calculator）
        
        Returns
        -------
        mmcalc : MinModeCalculator
            最小模式计算器实例
        """
        if self._mm_calc is None:
            self._mm_calc = MinModeCalculator(std_calc=atoms.calc)
        else:
            self._mm_calc.std_calc = atoms.calc
        return self._mm_calc

    def _calc_min_eigenvalue(self, atoms: Atoms) -> float:
        """
        计算Hessian矩阵的最小特征值
        
        用于判断优化收敛点是否为稳态（特征值>0）或鞍点（特征值<0）。
        这是一个相对昂贵的操作，使用有限差分法计算Hessian。
        
        Parameters
        ----------
        atoms : Atoms
            优化收敛后的原子结构
        
        Returns
        -------
        eig_val : float
            Hessian矩阵的最小特征值
        
        Notes
        -----
        每次调用前强制将 min_mode 重置为 None，迫使 Lanczos 从随机向量出发，
        避免复用上次（通常在盆地底部）算出的正曲率方向作为初猜，从而防止
        Krylov 子空间被限制在正交补空间、错过真实虚频方向、将鞍点误判为稳态。
        同理，调用后也清零 min_mode，杜绝跨调用的残留污染。
        注意：BasinManager._mm_calc 仅在本方法中使用，此处清零不影响
        HybridBoostCalculator._mm_calc 或其他模块的热启动逻辑。
        """
        mm_atoms = atoms.copy()
        mm_calc = self._get_mmcalc(atoms)
        # 强制从随机向量出发，防止继承上次盆地底部的正曲率 min_mode 作为初猜，
        # 导致 Lanczos 在鞍点处找不到真实负曲率方向而返回假正特征值。
        mm_calc.parameters["min_mode"] = None
        mm_atoms.calc = mm_calc
        mm_atoms.calc.calculate(mm_atoms)
        eig_val = mm_atoms.calc.parameters["eig_values"][0]
        
        # 累加 MinModeCalculator 的 fcalls
        self.fcalls += mm_calc.fcalls
        mm_calc.fcalls = 0
        
        # 清理临时引用，并二次清零 min_mode 防止残留污染下一次调用
        mm_atoms.calc = None
        if hasattr(mm_calc, 'atoms'):
            mm_calc.atoms = None
        if hasattr(mm_calc, 'results'):
            mm_calc.results = {}
        mm_calc.parameters["min_mode"] = None
        
        return eig_val

    def _clear_calc_atoms(self, calc):
        """
        清理calculator的内部atoms引用
        
        防止循环引用导致的内存泄漏。
        
        Parameters
        ----------
        calc : Calculator
            要清理的计算器
        """
        if calc is None:
            return
        if hasattr(calc, 'atoms'):
            calc.atoms = None
        if hasattr(calc, 'results'):
            calc.results = {}

    def _get_optimizer(self, atoms: Atoms):
        """
        获取或复用优化器实例
        
        首次调用时创建优化器，之后复用并重置内部状态以提高效率。
        
        Parameters
        ----------
        atoms : Atoms
            要优化的原子对象
        
        Returns
        -------
        optimizer : Optimizer
            优化器实例
        opt_atoms : Atoms
            优化器绑定的原子对象
        """
        self._opt_atoms = atoms.copy()
        self._opt_atoms.calc = atoms.calc
        self._optimizer = self._optimizer_class(self._opt_atoms, logfile=None)
        
        return self._optimizer, self._opt_atoms

    def _write_log_header(self, file_handle):
        """
        写入日志文件头部
        
        Parameters
        ----------
        file_handle : file object
            打开的文件句柄
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
        格式化写入单行日志
        
        Parameters
        ----------
        file_handle : file object
            打开的文件句柄
        step : int
            当前优化步数
        min_dist : float
            到最近稳态的距离
        reason : str or float
            终止原因（"FAST"=快速命中, "SADDLE"=鞍点, " "=正常）
        basin_id : int
            稳态ID（-1表示鞍点）
        energy : float
            当前能量
        fmax : float
            当前最大力
        eig_val : float or str
            Hessian最小特征值（" "表示未计算）
        """
        # 格式化reason
        if isinstance(reason, str):
            reason_str = f"{reason:>6s}"
        else:
            reason_str = f"{reason:>6.2f}"
        
        # 格式化eig_val
        if isinstance(eig_val, str):
            eig_str = f"{eig_val:>12s}"
        else:
            eig_str = f"{eig_val:>12.4f}"
        
        line = (
            f"{step:>6d} "
            f"{min_dist:>6.2f} "
            f"{reason_str} "
            f"{basin_id:>6d} "
            f"{energy:>12.4f} "
            f"{fmax:>12.4f} "
            f"{eig_str}\n"
        )
        file_handle.write(line)
