#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: Qian Lixiang
# @Email: 649811459@qq.com
__author__ = "Qian Lixiang"
__email__ = "649811459@qq.com"

"""
Bond-Boost超动力学计算模块

本模块实现Bond-Boost Hyperdynamics方法，通过监测原子键应变并施加偏置势来加速稀有事件采样。
与MMF方法相比，Bond-Boost不需要计算Hessian，计算效率更高，适合大规模系统和高温模拟。

核心原理：
1. 识别稳态并记录初始键长
2. MD过程中计算所有键的应变 ε_i = (r_i - r_i^0) / r_i^0
3. 找到最大应变键 ε_m
4. 施加偏置势 E_bias = A(ε_m) * Σ_i δV_i(ε_i)
   - δV_i: 单键的抛物线偏置势
   - A(ε_m): 包络函数，确保在接近过渡态时偏置势逐渐降为0

典型用法:
    >>> from ase.calculators.emt import EMT
    >>> from ase import Atoms
    >>> from ase.md import VelocityVerlet
    >>> 
    >>> # 创建Bond-Boost计算器
    >>> bb_calc = BondBoostCalculator(
    >>>     std_calc=EMT(),
    >>>     emax=0.5,  # 最大偏置能量(eV)
    >>>     cutoff=3.0,  # 键距离截断(Å)
    >>>     temperature_K=300
    >>> )
    >>> 
    >>> atoms = Atoms('Cu64', ...)
    >>> atoms.calc = bb_calc
    >>> 
    >>> # 运行超动力学MD
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
    Bond-Boost Hyperdynamics偏置势计算器
    
    通过监测原子键应变施加偏置势以加速稀有事件采样。相比MMF方法，Bond-Boost
    不需要计算Hessian矩阵，计算效率更高，特别适合大规模系统和高温模拟。
    
    算法流程：
    1. **稳态识别**：使用BasinManager识别并记录稳态结构
    2. **键拓扑构建**：根据cutoff参数建立键连接关系
    3. **应变计算**：对每个键计算应变 ε_i = (r_i - r_i^0) / r_i^0
    4. **偏置势计算**：
       - 单键偏置：δV_i = (emax/Nb) * (1 - (ε_i/q)^2) if |ε_i| < q, else 0
       - 总偏置基：V_b = Σ_i δV_i
       - 包络函数：A(ε_m) = (1-(ε_m/q)^2)^2 / (1-P1^2(ε_m/q)^2)
       - 总偏置势：E_bias = A(ε_m) * V_b
    5. **力计算**：通过链式法则计算偏置力 F_bias = -∇E_bias
    
    Parameters
    ----------
    std_calc : Calculator
        标准势能面计算器（如EMT, VASP等）
    
    logfile : str or file object, default='Bond.log'
        日志输出文件
    
    cutoff : float, default=3.0
        键距离截断（Å）。两个原子距离小于cutoff时认为有键连接。
        - 过小：遗漏部分键，可能导致偏置不足
        - 过大：包含过多非键相互作用，降低效率
        推荐设为第二近邻距离
    
    emax : float, default=0.5
        最大偏置能量（eV）。控制偏置势的强度。
        - 过小：加速效果不明显
        - 过大：可能引入非物理行为
        推荐范围：0.3 ~ 1.0 eV
    
    q : float, default=0.37
        应变截断参数。当|ε_i| > q时，该键的偏置势为0。
        - 较小值：偏置势在较小应变时就开始衰减（保守）
        - 较大值：允许较大应变（激进）
        推荐范围：0.3 ~ 0.5
    
    max_q : float, default=2.0
        最大应变阈值倍数。当ε_m > max_q*q时，认为已脱离稳态，需重新识别稳态。
    
    P1 : float, default=0.9
        包络函数参数。控制包络函数在ε_m接近q时的衰减速度。
        - 接近1：衰减缓慢
        - 接近0：衰减快速
        推荐范围：0.8 ~ 0.95
    
    delta : float, default=1e-6
        数值微分步长（保留参数，当前未使用）
    
    temperature_K : float, optional
        系统温度（K）。用于计算加速因子ACT = exp(E_bias/kT)。
    
    write_basins : bool, default=True
        是否将新发现的稳态写入'basins.traj'
    
    write_bias_log : bool, default=True
        是否将偏置能量、温度、ACT写入'bias.log'
    
    bias_interval : int, default=10
        bias.log输出间隔（每N次calculate调用输出一次）
    
    pbc_wrap : bool, default=True
        每次calculate后是否将原子包裹到主晶胞内（保留参数，当前未使用）
    
    Attributes
    ----------
    bias_energy : float
        当前的偏置能量（eV）
    
    bias_forces : np.ndarray, shape (n_atoms, 3)
        当前的偏置力（eV/Å）
    
    basin_id : int
        当前所处的稳态ID
    
    n_bonds : int
        当前稳态的键数量
    
    epsilon_m : float
        最大键应变
    
    Examples
    --------
    基本用法：
    
    >>> from ase.build import bulk
    >>> from ase.calculators.emt import EMT
    >>> from ase.md import VelocityVerlet
    >>> from ase import units
    >>> 
    >>> # 创建原子结构
    >>> atoms = bulk('Cu', 'fcc', a=3.6).repeat((4, 4, 4))
    >>> atoms.rattle(stdev=0.05)
    >>> 
    >>> # 创建Bond-Boost计算器
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
    >>> # 运行超动力学MD
    >>> dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)
    >>> for i in range(100):
    >>>     dyn.run(100)
    >>>     bias_E = bb_calc.parameters['bias_energy']
    >>>     eps_m = bb_calc.epsilon_m
    >>>     print(f"Step {i*100}: Bias={bias_E:.3f} eV, ε_m={eps_m:.4f}")
    
    调整参数以适应不同系统：
    
    >>> # 高温系统，增大emax
    >>> bb_calc = BondBoostCalculator(
    >>>     std_calc=EMT(),
    >>>     emax=0.8,
    >>>     q=0.4,
    >>>     temperature_K=600
    >>> )
    >>> 
    >>> # 低温系统，减小emax
    >>> bb_calc = BondBoostCalculator(
    >>>     std_calc=EMT(),
    >>>     emax=0.3,
    >>>     q=0.3,
    >>>     temperature_K=100
    >>> )
    
    Notes
    -----
    - calculate()返回的能量和力已包含偏置项
    - 键拓扑在每个稳态建立一次，MD过程中保持不变
    - 当max_q*q < ε_m时，自动尝试识别新稳态并更新键拓扑
    - 键长计算自动考虑周期性边界条件（最小镜像约定）
    - 偏置力通过链式法则精确计算，而非数值微分
    - BasinManager的详细输出写入'rlx.log'
    - Bond-Boost方法不保证时间可逆性，因此不适用于平衡态性质计算
    
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
        # 结果存储
        "basin_id": -1,
        "bias_energy": None,
        "bias_forces": None,
        "fcalls": 0,
    }

    def __init__(
        self,
        std_calc: Calculator,
        # 计算器参数
        cutoff: float = 3.0,
        emax: float = 0.5,
        q: float = 0.37,
        max_q: float = 2.0,
        P1: float = 0.9,
        # 其他参数
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

        # 稳态相关参数（每个稳态更新一次）
        self.positions0: Optional[np.ndarray] = None  # 稳态位置 (n_atoms, 3)
        self.bond_indices: np.ndarray  # 键索引 (n_bonds, 2)
        self.n_bonds: int = 0  # 键数量
        self.bond_lengths0: np.ndarray  # 稳态键长 (n_bonds,)

        # MD过程中更新的参数
        self.positions: np.ndarray  # 当前位置 (n_atoms, 3)
        self.bond_lengths: np.ndarray  # 当前键长 (n_bonds,)
        self.bond_vectors: np.ndarray  # 键向量 (n_bonds, 3)
        self.epsilons: np.ndarray  # 键应变 (n_bonds,)
        self.grad_epsilons_per_bond: np.ndarray  # 应变梯度 (n_bonds, 3)
        self.delta_potentials: np.ndarray  # 单键偏置势 (n_bonds,)
        self.Vb: float  # 总偏置势基数
        self.grad_delta_potentials: np.ndarray  # 偏置势梯度 (n_bonds,)
        self.epsilon_m_index: int  # 最大应变键索引
        self.epsilon_m: float = 0.0  # 最大应变
        self.grad_epsilon_m: np.ndarray  # 最大应变梯度 (3,)
        self.A: float  # 包络函数值
        self.grad_A: float  # 包络函数梯度
        self.bias_energy: float  # 偏置能量
        self.bias_forces: np.ndarray  # 偏置力 (n_atoms, 3)

    def _calc_pbc_bond_vectors(self, positions: np.ndarray) -> np.ndarray:
        """
        计算考虑周期性边界条件的键向量
        
        使用最小镜像约定计算键向量，确保得到最短的键距离。
        
        Parameters
        ----------
        positions : np.ndarray, shape (n_atoms, 3)
            原子位置
        
        Returns
        -------
        bond_vectors : np.ndarray, shape (n_bonds, 3)
            键向量
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

        # 转换为分数坐标
        cell_inv = np.linalg.inv(cell)
        scaled_diff = diff @ cell_inv

        # 应用最小镜像约定
        for i in range(3):
            if pbc[i]:
                scaled_diff[:, i] = scaled_diff[:, i] - np.round(
                    scaled_diff[:, i]
                )

        return scaled_diff @ cell

    def _calc_delta_potentials(self, epsilons: np.ndarray) -> np.ndarray:
        """
        计算单键偏置势
        
        δV_i = (emax/Nb) * (1 - (ε_i/q)^2) if |ε_i| < q, else 0
        
        Parameters
        ----------
        epsilons : np.ndarray, shape (n_bonds,)
            键应变
        
        Returns
        -------
        delta_potentials : np.ndarray, shape (n_bonds,)
            单键偏置势
        """
        emax = self.emax
        q = self._q
        Nb = self.n_bonds
        results = emax * (1 - (epsilons / q) ** 2) / Nb
        results[np.abs(epsilons) > q] = 0
        return results

    def _calc_grad_delta_potential(self, epsilons: np.ndarray) -> np.ndarray:
        """
        计算单键偏置势关于应变的梯度
        
        ∂(δV_i)/∂ε_i = -2*emax*ε_i / (Nb*q^2) if |ε_i| < q, else 0
        
        Parameters
        ----------
        epsilons : np.ndarray, shape (n_bonds,)
            键应变
        
        Returns
        -------
        gradients : np.ndarray, shape (n_bonds,)
            偏置势梯度
        """
        emax = self.emax
        q = self._q
        Nb = self.n_bonds
        results = -2 * emax * epsilons / q**2 / Nb
        results[np.abs(epsilons) > q] = 0
        return results

    def _calc_envelope(self, epsilon_m: float) -> float:
        """
        计算包络函数
        
        A(ε_m) = (1-(ε_m/q)^2)^2 / (1-P1^2(ε_m/q)^2) if |ε_m| < q, else 0
        
        包络函数确保在接近过渡态（ε_m接近q）时偏置势平滑降为0。
        
        Parameters
        ----------
        epsilon_m : float
            最大应变
        
        Returns
        -------
        A : float
            包络函数值
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
        计算包络函数关于最大应变的梯度
        
        ∂A/∂ε_m
        
        Parameters
        ----------
        epsilon_m : float
            最大应变
        
        Returns
        -------
        dA : float
            包络函数梯度
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

        # build bond topology from the *relaxed* basin positions
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

        # reference bond lengths (PBC-aware)
        bond_vectors0 = self._calc_pbc_bond_vectors(self.positions0)
        self.bond_lengths0 = np.linalg.norm(bond_vectors0, axis=1)

        self._log(f"# Basin {sid}: {self.n_bonds} bonds identified")

    def _update_positions(self):
        """更新当前原子位置"""
        self.positions = self.atoms.get_positions()

    def _update_bond_lengths(self):
        """更新当前键长"""
        self.bond_vectors = self._calc_pbc_bond_vectors(self.positions)
        self.bond_lengths = np.linalg.norm(self.bond_vectors, axis=1)

    def _update_epsilons(self):
        """更新键应变 ε_i = (r_i - r_i^0) / r_i^0"""
        self.epsilons = (self.bond_lengths - self.bond_lengths0) / self.bond_lengths0

    def _update_grad_epsilons(self):
        """
        更新应变梯度 ∂ε_i/∂r_α
        
        对于键i连接原子(a1, a2)：
        ∂ε_i/∂r_a1 = (r_a1 - r_a2) / (r_i^0 * r_i)
        ∂ε_i/∂r_a2 = -∂ε_i/∂r_a1
        """
        diff = self.bond_vectors  # (n_bonds, 3)
        prefactor = 1.0 / (self.bond_lengths0 * self.bond_lengths)
        prefactor = prefactor[:, np.newaxis]
        self.grad_epsilons_per_bond = diff * prefactor  # (n_bonds, 3)

    def _update_delta_potentials(self):
        """更新单键偏置势和总偏置势基数"""
        self.delta_potentials = self._calc_delta_potentials(self.epsilons)
        self.Vb = self.delta_potentials.sum()

    def _update_grad_delta_potentials(self):
        """更新单键偏置势梯度"""
        self.grad_delta_potentials = self._calc_grad_delta_potential(
            self.epsilons
        )

    def _update_epsilon_m(self):
        """
        更新最大应变及其梯度
        
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
        """更新包络函数值"""
        self.A = self._calc_envelope(self.epsilon_m)

    def _update_grad_envelope(self):
        """更新包络函数梯度"""
        self.grad_A = self._calc_grad_envelope(self.epsilon_m)

    def _update_bias(self):
        """
        更新偏置势能和偏置力
        
        偏置能量：E_bias = A(ε_m) * V_b
        偏置力：F_bias = -∇E_bias (通过链式法则计算)
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

        # 偏置能量
        self.bias_energy = self.A * self.Vb

        # 偏置力计算：F_bias = -A * Σ_i (∂δV_i/∂ε_i) * (∂ε_i/∂r)
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

        # unbiased energy & forces
        self._update_std_results()

        # ── check if the basin reference needs updating ──
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

        # ── compute bias ──
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

        # ── write bias.log ──
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
