# HDkit —— 超动力学示例套件

[English](README.md) | [简体中文](README.zh-cn.md)

一套轻量级、免安装的**超动力学（Hyperdynamics, HD）分子动力学**模拟工具包，
为 Cu(100) 表面扩散提供了统一的运行程序。

> **HDkit 是一篇投稿中论文的配套代码仓库。**
> 本仓库仅供复现和验证论文结果，**不适用于生产级模拟。**

> **操作系统**：仅支持 Linux 和 macOS。Windows 用户请先安装
> [WSL](https://learn.microsoft.com/zh-cn/windows/wsl/install)。

本仓库作为论文补充材料，提供以下内容：

- **`HDkit/`** —— 精简的 Python 包，包含核心 HD 计算器（Bond-Boost、MMF、BasinManager）
- **`run_compare.py`** —— 在同一结构上对比三种 HD 方法的偏置（单步评估）
- **`run_hd.py`** —— 使用 Bond-Boost、MMF 或 J-MMF 运行多步 HD-MD

> **注意**：HDkit 是论文算法的**简化参考实现**，聚焦于核心方法的正确性。
> 一些工程细节（错误恢复、MPI 支持、生产级 I/O）被刻意精简。
> 示例中的模拟时长（BB: 10 ns; MMF/J-MMF: 100 ps）可在几分钟到几小时内
> 验证代码能否从头跑到尾。论文中报告的结果需要在 HPC 资源上进行更长时间
> （µs 量级）的运行。

---

## 目录

- [快速开始](#快速开始)
  - [获取仓库](#获取仓库)
  - [环境配置](#环境配置)
  - [验证安装](#验证安装)
- [仓库结构](#仓库结构)
- [偏置对比 (`run_compare.py`)](#偏置对比-run_comparepy)
- [HD 模拟 (`run_hd.py`)](#hd-模拟-run_hdpy)
- [输出文件](#输出文件)
- [参考文献](#参考文献)
- [关于](#关于)
- [许可](#许可)

---

## 快速开始

### 获取仓库

**方式一 —— 下载 Release**（无需 Git）：

> [![Release](https://img.shields.io/github/v/release/ZhangLabTHU/HDkit-example?color=blue)](https://github.com/ZhangLabTHU/HDkit-example/releases)
>
> 前往 [Releases 页面](https://github.com/ZhangLabTHU/HDkit-example/releases)
> 下载 `HDkit-example-v1.0.0.zip` 或 `.tar.gz`，解压后进入目录即可使用。

**方式二 —— 通过 Git 克隆**：

```bash
git clone https://github.com/ZhangLabTHU/HDkit-example.git
cd HDkit-example
```

> **Results 分支**：自行运行模拟后，可切换到 `results` 分支查看预计算的输出：
>
> ```bash
> git checkout results
> ```

仓库包含运行示例所需的一切：

| 文件/目录 | 用途 |
|---|---|
| `HDkit/` | 核心 HD 计算器库（无需 `pip install`） |
| `run_hd.py` | 多步 HD-MD 运行器 |
| `run_compare.py` | 单步偏置对比 |
| `verify.py` | 环境验证脚本 |
| `hd-ini.traj` | HD 模拟初始结构 |
| `compare-ini.traj` | 偏置对比初始结构 |
| `Cu_u3.eam` | Cu EAM 势文件 |
| `CHANGELOG.md` | 发布说明 |
| `Makefile` | 构建 Release 包 (`make release`) |

> **无需安装**。`HDkit/` 位于项目根目录，从此目录运行脚本即可直接
> `import HDkit`，无需配置路径。

### 环境配置

| 依赖 | 最低版本 | 说明 |
|---|---|---|
| **Python** | ≥ 3.10 | 使用 `match/case` 语法 |
| **ASE** | ≥ 3.22 | 提供 `NoseHooverChainNVT` 积分器；自带 NumPy |
| **LAMMPS** | 需 Python 绑定 | EAM 势求解器 |

推荐使用独立的 conda 环境：

```bash
conda create -n HDkit -c conda-forge python=3.11 ase lammps -y
conda activate HDkit
```

### 验证安装

```bash
python verify.py
```

所有检查通过后即可运行示例。

---

## 仓库结构

```
.
├── HDkit/                          # ← 轻量级 HD 工具包（无需安装）
│   ├── __init__.py                 #   包初始化
│   ├── basin.py                    #   BasinManager：势能面势阱识别与持久化
│   └── calculators/
│       ├── __init__.py             #   计算器注册
│       ├── basecalculator.py       #   BaseCalculator：抽象基类
│       ├── minmode.py              #   MinModeCalculator：Hessian 最小模式（Lanczos）
│       ├── bondboost.py            #   BondBoostCalculator：Bond-Boost HD 方法
│       └── ridge/
│           ├── __init__.py
│           └── mmf.py              #   MMFPathCalculator：MMF 势脊 HD 方法
│
├── run_hd.py                       # ← 多步 HD-MD 运行器 (bb | mmf | j-mmf)
├── run_compare.py                  # ← 单步偏置对比
├── verify.py                       #   安装验证脚本
├── compare-ini.traj                #   偏置对比用初始结构
├── hd-ini.traj                     #   HD-MD 用初始结构
├── Cu_u3.eam                       #   Cu EAM 势文件
├── README.md                       #   英文文档
├── README.zh-CN.md                 #   中文文档
├── CHANGELOG.md                    #   发布说明
└── Makefile                        #   构建 Release 包 (`make release`)
```

运行后，输出文件写入对应子目录（`Climb/`、`Bond-Boost/`、`MMF/` 或 `J_MMF/`）。

### HDkit 模块

| 模块 | 类 | 用途 |
|---|---|---|
| `basin.py` | `BasinManager` | 通过结构优化识别和缓存势阱（局部极小值）；通过 Hessian 特征值分析检测鞍点 |
| `calculators/basecalculator.py` | `BaseCalculator` | 抽象基类，提供 `std_calc`（无偏势能面）接口和日志 |
| `calculators/minmode.py` | `MinModeCalculator` | 通过 Lanczos 迭代或完全对角化计算 Hessian 最小特征向量；MMF 用于确定爬升方向 |
| `calculators/bondboost.py` | `BondBoostCalculator` | Bond-Boost 方法：监测键应变，施加抛物线型偏置，在过渡态附近包络函数归零 |
| `calculators/ridge/mmf.py` | `MMFPathCalculator` | MMF 方法：沿最小模式方向爬升定位能量势脊，在鞍点区域施加偏置。支持 Simple 和 Shear Jacobian 算法 |

---

## 偏置对比（`run_compare.py`）

`run_compare.py` 对**同一初始结构**（`compare-ini.traj`）分别应用三种 HD 方法，
输出每种方法产生的偏置能量和力的大小。这是一次**单步评估**（不执行 MD 积分），
方便你快速对比三种方法对同一原子构型的响应。

```bash
python run_compare.py
```

输出写入 `Climb/` 目录：

| 文件 | 内容 |
|---|---|
| `compare-ini.traj` | 含 std_calc 能量和力的初始结构 |
| `hyper-bb.traj` | Bond-Boost — 总（有偏）能量和力 |
| `hyper-mmf.traj` | MMF Simple — 总能量和力 |
| `hyper-j-mmf.traj` | J-MMF Shear — 总能量和力 |
| `bias-bb.traj` | Bond-Boost — 仅偏置能量和力 |
| `bias-mmf.traj` | MMF Simple — 仅偏置能量和力 |
| `bias-j-mmf.traj` | J-MMF Shear — 仅偏置能量和力 |
| `climb-mmf.traj` | MMF Simple — 完整爬升路径轨迹 |
| `climb-j-mmf.traj` | J-MMF Shear — 完整爬升路径轨迹 |

MMF 方法使用 `emax = −1`（无能量上限），爬升可到达势脊。
`hyper-` 轨迹存储完整的有偏能量和力；`bias-` 轨迹仅存储偏置贡献，
适合使用标准能量/力数组的可视化工具。

**诊断日志** —— `run_compare.py` 开启详细输出，便于检查每一步计算：

| 日志 | 内容 |
|---|---|
| `rlx.log` | BasinManager — 势阱识别中的每步优化 |
| `climb.log` | MMF — 每次爬升步骤（能量、势阱 ID、耗时） |
| `mode.log` | MinModeCalculator — Lanczos 迭代和收敛角度 |
| `Bond.log` | BondBoost — 势阱更新通知 |

相比之下，`run_hd.py` 运行时会**关闭详细输出**，避免长时间 MD 产生海量日志。

---

## HD 模拟（`run_hd.py`）

`run_hd.py` 从 `hd-ini.traj` 开始执行完整的超动力学 MD 模拟，演示了三种方法的
完整工作流 —— 平衡、偏置加速生产、后处理。

> **运行前**：激活 Python 环境并确保 LAMMPS 可访问。
>
> ```bash
> conda activate HDkit
> ```

所有示例从**项目根目录**启动。无需 `pip install` 或路径配置，
脚本和 `HDkit/` 在同一目录，`import HDkit` 直接可用。

```bash
python run_hd.py bb        # Bond-Boost
python run_hd.py mmf       # MMF Simple (J_algo="s")
python run_hd.py j-mmf     # J-MMF Shear  (J_algo="h", 推荐)
```

方法名**不区分大小写**（`BB`、`bb`、`Bb` 均可）。
输出写入对应子目录（`Bond-Boost/`、`MMF/` 或 `J_MMF/`）。

### 方法参数对比

| 参数 | 方法 | emax | 生产时长 | loginterval | 特点 |
|---|---|---|---|---|---|
| `bb` | Bond-Boost | 0.3 eV | 10 ns | 10000 | ~1 次力计算/步，非常高效 |
| `mmf` | MMF Simple | 0.5 eV | 100 ps | 100 | 直接使用势脊力（基线方法） |
| `j-mmf` | J-MMF Shear | 0.5 eV | 100 ps | 100 | Jacobian 传播 + 正交投影 |

所有运行使用 500 K、1 fs 时间步长、Nose–Hoover chain NVT 热浴、
10 ps 无偏平衡，然后切换至 HD 计算器。

> **模拟时长说明**：默认设置兼顾了周转时间和统计质量。
> Bond-Boost 运行 10 ns 是因为计算成本很低（每步约 1 次力计算），
> MMF 方法运行 100 ps 是因为每步涉及多次力评估（爬升、Hessian 对角化）。
> 所有时长可通过 `run_hd.py` 中的 `prod_steps` 调整。
> 论文中的生产结果需要在 HPC 资源上进行更长时间（ns–µs 量级）的运行。

### 模拟工作流

1. 将 `hd-ini.traj` 和 `Cu_u3.eam` 复制到输出目录
2. 读取结构 → 设置 LAMMPS EAM 计算器（无偏势能面）
3. 初始化 Maxwell–Boltzmann 速度
4. 平衡阶段：500 K 下 NVT 10 ps（使用无偏 std_calc）
5. 生产阶段：NVT HD-MD —— 10 ns（BB）或 100 ps（MMF / J-MMF）
6. 后处理：提取势阱跃迁 → 计算 ACT
7. 转换 `hd.traj` → `bias_hd.traj`（仅偏置轨迹，用于可视化）

---

## 输出文件

### `run_hd.py` 输出（在 `Bond-Boost/`、`MMF/` 或 `J_MMF/` 目录中）

长时间 MD 运行时日志输出**最小化**。

| 文件 | 说明 |
|---|---|
| `bias.log` | 每步偏置能量、温度和 ACT（加速因子） |
| `basins.traj` | ASE 轨迹文件，记录识别的势阱（稳态）结构 |
| `basins.log` | 势阱跃迁汇总（帧号、距离、移动原子数） |
| `hd.traj` | 完整 MD 轨迹 |
| `bias_hd.traj` | 仅偏置轨迹（每帧的偏置能量和力） |
| `HD.log` | ASE MD 日志（能量、温度等） |
| `fin.traj` | 最终原子构型 |
| `ini-T.traj` | 平衡后的结构 |
| `basin.pkl` | Pickle 格式的 BasinManager 数据库（用于重启） |
| `rlx.log` | BasinManager — 每次势阱识别的最终结果行 |
| `Bond.log` | BB 内部日志（仅 Bond-Boost） |
| `climb.log` | MMF 爬升日志（仅 MMF/J-MMF，仅输出最终结果） |
| `mode.log` | Lanczos 头行（仅 MMF/J-MMF） |

### `run_compare.py` 输出（在 `Climb/` 目录中）

日志输出**详细**——记录每一步优化、爬升迭代和 Lanczos 收敛检查。

| 文件 | 说明 |
|---|---|
| `hyper-bb.traj` / `hyper-mmf.traj` / `hyper-j-mmf.traj` | 总（有偏）能量和力 |
| `bias-bb.traj` / `bias-mmf.traj` / `bias-j-mmf.traj` | 仅偏置能量和力 |
| `climb-mmf.traj` / `climb-j-mmf.traj` | 完整爬升路径轨迹（仅 MMF） |
| `compare-ini.traj` | 含 std_calc 能量和力的初始结构 |
| `rlx.log` | BasinManager — 每次势阱识别的每一步优化 |
| `climb.log` | MMF — 每次爬升步骤（步号、能量、势阱 ID、时间） |
| `mode.log` | MinModeCalculator — Lanczos 迭代细节和收敛情况 |
| `Bond.log` | BondBoost — 势阱更新通知 |

### 关键指标

- **ACT**（Accelerated Corrected Time，加速校正时间）：$\text{ACT} = \exp(\Delta V / k_BT)$，瞬态时间加速因子。
- **HD time**（超动力学时间）：$\langle\text{ACT}\rangle \times t_\text{wall}$，考虑偏置后的有效模拟时间。

---

## 参考文献

1. Voter, A. F. Hyperdynamics: Accelerated molecular dynamics of infrequent
   events. *Phys. Rev. Lett.* **78**, 3908–3911 (1997).
2. Miron, R. A. & Fichthorn, K. A. Accelerated molecular dynamics with the
   Bond-Boost method. *J. Chem. Phys.* **119**, 6210–6216 (2003).
3. Xiao, P., Duncan, J., Zhang, L. & Henkelman, G. Ridge-based bias potentials
   to accelerate molecular dynamics. *J. Chem. Phys.* **143**, 244104 (2015).

---

## 关于

### 项目背景

**HDkit** 是从 **[DLTS](https://github.com/ZhangLabTHU/Hyperdynamics)**
（Deep Long Time Simulation package，深度长时模拟包）中提取的轻量级超动力学工具包。
DLTS 由 [ZhangLab](https://www.zhanglab-thu.com) 开发，面向长时间尺度动力学模拟，
涵盖分子动力学（Hyperdynamics）和自适应动力学蒙特卡洛（aKMC）等多个方向。
HDkit 仅包含 DLTS 中与超动力学相关的代码，并做了精简以便使用和复现。

- **ZhangLab 主页**：<https://www.zhanglab-thu.com>
- **ZhangLab GitHub**：[@ZhangLabTHU](https://github.com/ZhangLabTHU)

### 作者

HDkit 和 DLTS 中的超动力学相关代码由
**[PhoenixQian](https://github.com/PhoenixQian)** 编写。

- **邮箱**：[649811459@qq.com](mailto:649811459@qq.com)
- **GitHub**：<https://github.com/PhoenixQian>

### 论文状态

本仓库为一篇投稿中论文的配套代码。论文链接和推荐引用格式将在发表后更新于此。

---

## 许可

本代码**仅用于验证和复现配套论文的结果**。这是一个简化的参考实现，
**不适用于生产级分子动力学模拟**。如果工作中使用了本工具包，请引用上述参考文献。
