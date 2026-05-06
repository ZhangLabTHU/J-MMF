#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: Qian Lixiang
# @Email: 649811459@qq.com
__author__ = "Qian Lixiang"
__email__ = "649811459@qq.com"

"""
Hessian最小模式计算模块

本模块提供MinModeCalculator类，用于计算原子系统Hessian矩阵的最小特征值和特征向量。
主要用于识别势能面逃逸方向，是超动力学方法（如ridge-based）的核心组件。

支持两种计算方法：
1. Lanczos迭代法：高效，适合大规模系统，支持热启动加速收敛
2. Vibration全Hessian法：精确，适合小规模系统或Lanczos不收敛时的备用方案

典型用法:
    >>> from ase.calculators.emt import EMT
    >>> from ase import Atoms
    >>> 
    >>> atoms = Atoms('Cu4', ...)
    >>> atoms.calc = EMT()
    >>> 
    >>> # 使用Lanczos方法计算最小模式
    >>> mmcalc = MinModeCalculator(std_calc=EMT(), algo='Lanczos')
    >>> atoms.calc = mmcalc
    >>> forces = atoms.get_forces()  # 返回最小模式方向
    >>> 
    >>> # 获取特征值
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
    归一化向量
    
    Parameters
    ----------
    v : np.ndarray
        输入向量
    
    Returns
    -------
    normalized : np.ndarray
        归一化后的向量。如果输入为零向量，返回原向量。
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def vectors_angles(v1: np.ndarray, v2: np.ndarray, minimum: bool = True) -> np.ndarray:
    """
    计算两组向量之间的夹角
    
    支持单个向量或向量组的夹角计算，返回角度以度为单位。
    
    Parameters
    ----------
    v1, v2 : np.ndarray
        输入向量或向量组。形状可以是：
        - 1D: (n,) 单个向量
        - 2D: (n, m) m个n维向量
    minimum : bool, default=True
        是否返回最小夹角（考虑向量反向）。
        - True: 返回 min(angle, 180-angle)
        - False: 返回原始夹角 [0, 180]
    
    Returns
    -------
    angles : np.ndarray
        夹角数组（度）
    
    Raises
    ------
    ValueError
        如果两个输入的形状不匹配
    
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
    
    # 安全处理零向量情况
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
    Hessian最小模式计算器
    
    计算原子系统Hessian矩阵的最小特征值和对应特征向量，用于识别势能面的
    "最容易逃逸"方向。这是Min-Mode Following等超动力学方法的核心组件。
    
    支持两种计算算法：
    
    1. **Lanczos迭代法** (推荐)
       - 原理：通过Lanczos迭代构建三对角矩阵，求解其最小特征值
       - 优点：计算效率高，内存占用小，适合大规模系统
       - 支持热启动：可使用上一步的最小模式作为初始向量加速收敛
       - 收敛判据：特征向量旋转角度 < tolerance
    
    2. **Vibration全Hessian法** (备用)
       - 原理：通过有限差分计算完整Hessian矩阵并对角化
       - 优点：精确，不依赖迭代收敛
       - 缺点：计算量大，内存占用高
       - 使用场景：小系统或Lanczos不收敛时
    
    算法自动回退机制：
    - Lanczos不收敛时，自动尝试从随机向量重启
    - 随机向量仍不收敛，自动切换到Vibration方法
    
    Parameters
    ----------
    std_calc : Calculator
        标准势能面计算器（如EMT, VASP等），用于计算能量和力
    
    algo : str, default='Lanczos'
        最小模式计算算法：
        - 'Lanczos', 'lan', 'l': Lanczos迭代法
        - 'Vibration', 'vib', 'v': 全Hessian对角化法
    
    logfile : str or file object, default='mode.log'
        日志输出文件。设置为None使用os.devnull禁用日志。
    
    n_eigs : int, default=1
        计算的最小特征值数量。通常设为1即可。
        注意：n_eigs不能超过非约束原子的自由度数(3*n_unconstrained_atoms)
    
    indices : list of int, optional
        参与计算的原子索引列表。如果为None，自动选择所有非约束原子。
        用于减少计算量，例如只考虑表面原子。
    
    delta : float, default=1e-4
        有限差分位移步长（Å）。影响Hessian计算精度。
        - 过大：精度不足
        - 过小：数值误差增大
        推荐范围：1e-5 ~ 1e-3
    
    tolerance : float, default=1e-2
        Lanczos收敛容差（度）。当相邻迭代的特征向量旋转角度小于此值时认为收敛。
        - 较小值：更精确，但需要更多迭代
        - 较大值：更快，但精度降低
        推荐范围：1e-3 ~ 1e-1
    
    direction : {'central', 'forward', 'backward'}, default='central'
        有限差分方案：
        - 'central': (f(x+δ) - f(x-δ)) / 2δ，精度高但计算量加倍
        - 'forward': (f(x+δ) - f(x)) / δ
        - 'backward': (f(x) - f(x-δ)) / δ
    
    max_niter : int, default=200
        Lanczos最大迭代次数。如果未收敛，尝试重启或切换算法。
    
    min_mode : np.ndarray, optional
        上一步的最小模式向量，用于Lanczos热启动。
        - None: 从随机向量开始
        - 非None: 从该向量开始（加速收敛）
        通过 calc.parameters['min_mode'] = vector 设置
    
    Attributes
    ----------
    fcalls : int
        累计力计算次数，用于性能监控
    
    Examples
    --------
    基本用法：
    
    >>> from ase.build import bulk
    >>> from ase.calculators.emt import EMT
    >>> 
    >>> # 创建原子结构
    >>> atoms = bulk('Cu', 'fcc', a=3.6)
    >>> atoms.rattle(stdev=0.1)
    >>> 
    >>> # 创建最小模式计算器
    >>> mmcalc = MinModeCalculator(
    >>>     std_calc=EMT(),
    >>>     algo='Lanczos',
    >>>     tolerance=1e-2,
    >>>     logfile='mode.log'
    >>> )
    >>> 
    >>> # 计算最小模式
    >>> atoms.calc = mmcalc
    >>> min_mode_forces = atoms.get_forces()  # 返回最小模式方向向量
    >>> 
    >>> # 获取特征值
    >>> eig_val = mmcalc.parameters['eig_values'][0]
    >>> print(f"Minimum eigenvalue: {eig_val:.6f}")
    >>> 
    >>> # 检查是否为稳态（特征值>0）或鞍点（特征值<0）
    >>> if eig_val > 0:
    >>>     print("This is a stable minimum")
    >>> else:
    >>>     print("This is a saddle point")
    
    热启动加速收敛：
    
    >>> # 第一步：正常计算
    >>> mmcalc.parameters['min_mode'] = None
    >>> atoms.calc = mmcalc
    >>> _ = atoms.get_forces()
    >>> 
    >>> # 保存最小模式
    >>> saved_mode = mmcalc.parameters['min_mode']
    >>> 
    >>> # 第二步：使用上一步结果热启动
    >>> atoms2 = atoms.copy()
    >>> atoms2.rattle(stdev=0.05)
    >>> mmcalc.parameters['min_mode'] = saved_mode
    >>> atoms2.calc = mmcalc
    >>> _ = atoms2.get_forces()  # 收敛更快
    
    计算多个最小特征值：
    
    >>> mmcalc = MinModeCalculator(
    >>>     std_calc=EMT(),
    >>>     n_eigs=3  # 计算3个最小特征值
    >>> )
    >>> atoms.calc = mmcalc
    >>> _ = atoms.get_forces()
    >>> eig_vals = mmcalc.parameters['eig_values']
    >>> print(f"3 smallest eigenvalues: {eig_vals}")
    
    Notes
    -----
    - calculate()方法返回的forces是最小模式方向向量（归一化），不是真实的力
    - 如果 F0·min_mode < 0，返回 min_mode；否则返回 -min_mode（确保沿力的方向）
    - Hessian计算使用有限差分，计算量与原子数和delta相关
    - 对于大系统(>100原子)，强烈推荐使用Lanczos方法
    - 内存优化：大数组在使用后显式删除并调用gc.collect()
    - 与MPI并行计算器（如VASP）完全兼容，不使用文件缓存
    
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
        # 结果存储
        "min_mode": None,                                       # (N, 3)
        "eig_values": np.array(None),                           # (n_eigs,)
        "eig_vectors": np.array(None),                          # (3N, n_eigs)
        "fcalls": 0,
    }

    def __init__(
        self,
        std_calc: Calculator,
        # 计算器设置
        algo: str = "Lanczos",                                  # minmode计算算法
        n_eigs: int = 1,                                        # 计算的最小特征值数量
        indices: Optional[List[int]] = None,                    # 参与计算的原子索引
        delta: float = 1e-4,                                    # 有限差分步长
        # Lanczos设置
        tolerance: float = 1e-2,                                # Lanczos收敛容差（度）
        direction: str = "central",                             # 有限差分方案
        orth: Optional[bool] = None,                            # 是否完全正交化, None自动判断
        max_niter: int = 100,                                   # Lanczos最大迭代次数
        # 其他设置
        logfile: Optional[Union[str, object]] = "mode.log",     # 日志文件
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
        获取参与计算的原子索引列表
        
        如果用户未指定indices参数，自动选择所有非约束原子。
        
        Returns
        -------
        indices : list of int
            参与计算的原子索引列表
        """
        if self._indices is None:
            self._update_indices()

        return self._indices

    def _update_indices(self) -> None:
        """
        更新参与计算的原子索引列表
        """
        atoms = self.atoms.copy()
        n_atoms = len(atoms)

        # 自动选择非约束原子
        unconstrained_indices = {
            i for i in range(n_atoms)
            if not any(i in c.index for c in atoms.constraints)
        }
        indices = list(unconstrained_indices)
        self._indices = indices

    def _get_mask(self) -> np.ndarray:
        """
        获取原子掩码
        
        Returns
        -------
        mask : np.ndarray, shape (n_atoms,), dtype=bool
            True表示原子参与计算，False表示被约束
        """
        mask = np.ones(self.n_atoms, bool)
        mask[self._get_indices()] = False
        return ~mask

    def _calculate_partial_forces(self, q: np.ndarray) -> np.ndarray:
        """
        计算Hessian-向量乘积 H*q
        
        使用有限差分近似计算 df/dx，等价于Hessian矩阵与向量q的乘积。
        这是Lanczos迭代的核心操作，避免了显式构建完整Hessian矩阵。
        
        Parameters
        ----------
        q : np.ndarray, shape (3*n_atoms,)
            方向向量（扁平化）
        
        Returns
        -------
        H_q : np.ndarray, shape (3*n_atoms,)
            Hessian-向量乘积
        
        Notes
        -----
        - 约束原子的分量自动设为0
        - 有限差分方案由direction参数控制
        - 每次调用增加1-2次力计算（取决于direction）
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
                # 中心差分：精度高但计算量加倍
                atoms.set_positions(pos + displacement * delta)
                f_plus = atoms.get_forces().flatten()
                atoms.set_positions(pos - displacement * delta)
                f_minus = atoms.get_forces().flatten()
                self.fcalls += 2
                df = (f_minus - f_plus) / (2 * delta)
            case "forward":
                # 前向差分
                atoms.set_positions(pos + displacement * delta)
                f_plus = atoms.get_forces().flatten()
                self.fcalls += 1
                df = (f0 - f_plus) / delta
            case "backward":
                # 后向差分
                atoms.set_positions(pos - displacement * delta)
                f_minus = atoms.get_forces().flatten()
                self.fcalls += 1
                df = (f_minus - f0) / delta
            case _:
                raise ValueError(f"Unknown direction: {self._direction}")

        return df

    def _get_min_modes_lanczos(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用Lanczos迭代法计算最小模式
        
        Lanczos算法通过迭代构建Hessian的Krylov子空间，将其投影到三对角矩阵，
        从而高效求解最小特征值。相比完整对角化，计算量大幅减少。
        
        算法流程：
        1. 初始化：从min_mode（热启动）或随机向量开始
        2. Lanczos迭代：
           - 计算 v = H*q - beta*q_prev
           - 正交化（可选）
           - 更新三对角矩阵T
           - 对T求特征值，检查收敛
        3. 收敛判据：特征向量旋转角度 < tolerance
        4. 失败处理：重启或切换到Vibration方法
        
        Parameters
        ----------
        atoms : Atoms
            原子对象
        n_eigs : int, default=1
            计算的最小特征值数量
        
        Returns
        -------
        eig_values : np.ndarray, shape (n_eigs,)
            最小的n_eigs个特征值
        eig_vectors : np.ndarray, shape (3*n_atoms, n_eigs)
            对应的特征向量
        
        Notes
        -----
        - 内存优化：预分配所有大数组，及时清理临时变量
        - 如果从热启动向量收敛失败，自动尝试随机向量
        - 如果随机向量仍失败，回退到Vibration方法
        - 正交化仅在n_eigs>1时启用，避免不必要开销
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

        # 初始化：随机向量或热启动
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

        # 预分配数组
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

        # Lanczos主循环
        for k in range(1, max_niter + 1):
            q = Q[:, k - 1]
            v = self._calculate_partial_forces(q)
            v -= beta[k - 1] * q_prev
            alpha_k = q.T @ v
            alpha[k - 1] = alpha_k

            # 正交化（仅n_eigs>1时）
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

            # 更新三对角矩阵T
            T[k - 1, k - 1] = alpha_k
            if k > 1:
                T[k - 2, k - 1] = beta[k - 1]
                T[k - 1, k - 2] = beta[k - 1]

            # 检查收敛（k >= n_eigs时开始）
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

        # 输出特征值
        eigvalues_str = "".join(f"{eig_value:>14.4E}" for eig_value in eig_values[:self._n_eigs])
        self._log(f"eig_vals: {eigvalues_str}")

        # 检查特征值是否异常大
        if abs(eig_values[0]) > 1e5:
            self._log("Eigenvalue too large, convergence failed")
            converged = False

        # 处理结果
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
                # 热启动失败，尝试随机向量
                self._log("Restarting from random vector")
                self._log("=" * 10)
                self.parameters["min_mode"] = None
                return self._get_min_modes_lanczos()
            else:
                # 随机向量也失败，回退到Vibration方法
                self._log("Random vector failed. Falling back to Vibration method")
                self._log("=" * 10)
                return self._get_min_modes_vib()

    def _get_min_modes_vib(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用完整Hessian对角化计算最小模式

        通过有限差分计算完整Hessian矩阵，然后对角化得到所有特征值和特征向量。
        相比ASE Vibrations类，本实现避免了文件缓存，与MPI并行计算器完全兼容。
        
        算法流程：
        1. 对每个非约束原子的每个方向(x,y,z)进行位移
        2. 使用中心差分计算Hessian列
        3. 确保Hessian对称性（平均化）
        4. 对角化得到特征值和特征向量
        5. 将特征向量扩展到全系统（约束原子设为0）
        
        Parameters
        ----------
        atoms : Atoms
            原子对象
        n_eigs : int, default=1
            计算的最小特征值数量
        
        Returns
        -------
        eig_values : np.ndarray, shape (n_eigs,)
            最小的n_eigs个特征值
        eig_vectors : np.ndarray, shape (3*n_atoms, n_eigs)
            对应的特征向量（全系统）
        
        Notes
        -----
        - 计算量：每个原子需要6次力计算（3个方向×正负位移）
        - 内存占用：O(n_unconstrained^2)
        - 适用场景：小系统(<50原子)或Lanczos不收敛时
        - 与VASP等MPI计算器完全兼容，不使用文件缓存
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

        # 获取参考力
        pos0 = atoms.get_positions()
        f0 = self.std_results["forces"].flatten()

        # 初始化Hessian矩阵（仅非约束原子）
        n_dof = n_indices * 3
        hessian = np.zeros((n_dof, n_dof))

        # 计算Hessian矩阵的每一列
        for i, atom_idx in enumerate(indices):
            for j in range(3):  # x, y, z
                # 正向位移
                atoms.positions[atom_idx, j] = pos0[atom_idx, j] + delta
                f_plus = atoms.get_forces().flatten()
                self.fcalls += 1

                # 负向位移
                atoms.positions[atom_idx, j] = pos0[atom_idx, j] - delta
                f_minus = atoms.get_forces().flatten()
                self.fcalls += 1

                # 恢复位置
                atoms.positions[atom_idx, j] = pos0[atom_idx, j]

                # 计算Hessian列（中心差分）
                df_full = (f_minus - f_plus) / (2.0 * delta)
                df_reduced = df_full.reshape(-1, 3)[mask].flatten()
                hessian[:, i * 3 + j] = df_reduced

                if (i * 3 + j + 1) % 3 == 0:
                    self._log(f"Progress: {i+1}/{n_indices} atoms")

        # 对称化Hessian（提高数值精度）
        hessian = 0.5 * (hessian + hessian.T)

        # 对角化
        self._log("Diagonalizing Hessian matrix...")
        eig_values, eig_vectors_2d = np.linalg.eigh(hessian)

        # 诊断信息
        self._log(f"Hessian shape: {hessian.shape}")
        self._log(f"Condition number: {np.linalg.cond(hessian):.2e}")
        eigvalues_str = "".join(f"{eig_value:>14.4E}" for eig_value in eig_values[:min(3, self._n_eigs)])
        self._log(f"Smallest {min(3, self._n_eigs)} eigenvalues: {eigvalues_str}")

        # 归一化特征向量
        eig_vectors = []
        for i in range(self._n_eigs):
            eig_vectors.append(vunit(eig_vectors_2d[:, i]))
        eig_vectors = np.array(eig_vectors).T

        self._log("=" * 10)

        # 扩展到全系统（约束原子为0）
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
