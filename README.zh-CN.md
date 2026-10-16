<div align="center">

<h1>X2Real</h1>

<h3>面向真实世界通用机器人策略的广泛仿真评测基准</h3>

<p><strong>忠实 · 多样 · 公平 · 持续演进</strong></p>

<p>
  <a href="README.md">English</a> ·
  <a href="https://x2robot.com/en/pages/x2real">项目</a> ·
  <a href="https://arxiv.org/abs/2609.27449">论文</a> ·
  <a href="https://huggingface.co/datasets/x-square-robot/x2real-assets">资产</a> ·
  <a href="https://huggingface.co/datasets/x-square-robot/X2Real-Dataset">数据集</a> ·
  <a href="#news-zh">新闻</a> ·
  <a href="#highlights-zh">亮点</a> ·
  <a href="#documentation-zh">文档</a> ·
  <a href="#evaluation-suite-zh">评测</a>
</p>

<p>
  <a href="https://x2robot.com/en/pages/x2real"><img alt="X2Real 项目页" src="https://img.shields.io/badge/X2Real-Project_Page-0F766E?style=flat-square" height="22"/></a>&nbsp;
  <a href="https://arxiv.org/abs/2609.27449"><img alt="X2Real 论文" src="https://img.shields.io/badge/Paper-arXiv%3A2609.27449-B31B1B?style=flat-square" height="22"/></a>&nbsp;
  <a href="https://github.com/isaac-sim/IsaacSim/tree/release/6.0"><img alt="Isaac Sim 6.0.1" src="https://img.shields.io/badge/Isaac_Sim-6.0.1-475569?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=64748B" height="22"/></a>&nbsp;
  <a href="https://github.com/isaac-sim/IsaacLab/tree/release/3.0.0"><img alt="Isaac Lab 3.0" src="https://img.shields.io/badge/Isaac_Lab-3.0-475569?style=flat-square&labelColor=2563EB" height="22"/></a>&nbsp;
  <a href="https://github.com/isaac-sim/IsaacLab-Arena"><img alt="IsaacLab Arena" src="https://img.shields.io/badge/IsaacLab-Arena-475569?style=flat-square&labelColor=7C3AED" height="22"/></a>&nbsp;
  <a href="https://www.python.org/downloads/release/python-3120/"><img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-475569?style=flat-square&logo=python&logoColor=white&labelColor=6F5BD6" height="22"/></a>&nbsp;
  <a href="https://www.kernel.org/"><img alt="Linux 平台" src="https://img.shields.io/badge/Platform-Linux-475569?style=flat-square&logo=linux&logoColor=white&labelColor=334155" height="22"/></a>
</p>

</div>

<p align="center">
  <img src="docs/assets/task_final_1_4k.png" width="88%" alt="X2Real 任务体系">
</p>

<p align="center"><em>X2Real 任务体系。论文定义了 10 个核心能力维度下的 44 个分层长时序任务。</em></p>

<a id="news-zh"></a>

## 📰 新闻

**2026-10-10：** **X2Real v0.9.0** 正式开源，包含 ID case 评测运行时和发布文档。

<a id="highlights-zh"></a>

## ✨ 核心亮点

<p align="center">
  <img src="docs/assets/pipeline.png" width="100%" alt="Mana 任务设计与 rollout 分析流程">
</p>

<p align="center"><em>Mana 系统概览。Physical DSL 通过两个迭代循环连接任务设计、环境编译、策略 rollout 与轨迹评测。</em></p>

> X2Real 提供 ID 评测和人工示范数采。ManaEnv 负责环境构建、机器人控制、任务执行、
> 评分和结果产物；Policy Space v2 服务独立运行策略推理。键盘、Vuer/WebXR 和主从臂数采无需策略服务。

- 🎯 **忠实的仿真—实机评测** — 通过视觉、物理与底层控制标定对齐仿真和真实机器人；
  论文在所报告的实验设置下得到 `0.84` 的仿真—实机评测分数线性相关系数。
- 🧭 **十个能力维度** — 44 个推理、操作和挑战任务，将原子技能组合成分层长时序行为。
- 🎲 **多轴多样性** — 沿外观、任务设置和机器人本体随机化资产、布局、指令、相机、
  夹爪与桌面几何。
- 🤖 **五个正式 ID board** — 73 个可执行配置，覆盖 ArtiXon Arm-6A、Quanta X1、
  ARX R5 与 Franka 平台。
- 🧩 **可演进的基准基础设施** — Mana 的 Physical DSL 将场景、Task DAG、机器人本体
  和随机化规则组合成可执行环境。
- 🔌 **隔离的策略运行时** — 通过版本化 Policy Space 契约连接策略，无需把模型框架或
  checkpoint 放进评测器。
- 📊 **可复现的结果产物** — 默认确定性种子、board 预算、物化套件、结构化报告和
  episode 录制。

<a id="documentation-zh"></a>

## 📚 文档导航

| 章节 | 内容 |
| :-- | :-- |
| [安装与评测](#install-and-evaluation-zh) | 安装 ManaEnv、启动 Policy Space 并运行评测。 |
| [评测套件](#evaluation-suite-zh) | Board 和 ID case 数量。 |
| [仓库结构](#repository-structure-zh) | 关键文件和目录。 |
| [输出与评分](#outputs-and-scoring-zh) | 报告、视频和续跑。 |
| [人工遥操作数采](#manual-collection-zh) | 键盘、Vuer/WebXR、主从臂与 HDF5 示范数据。 |
| [范围](#release-scope-zh) | 内容范围。 |
| [论文引用](#citation-zh) | 引用 X2Real 论文。 |
| [许可证](#license-zh) | 许可证状态。 |

<a id="install-and-evaluation-zh"></a>

## 🚀 安装与评测

- Policy Space v2 提供策略端点。
- ManaEnv 负责环境、任务、评分和结果产物。

### 环境要求

- Linux、NVIDIA GPU，以及 Isaac Sim `6.0.1.0` 支持的驱动
- Python `3.12`，使用 Conda 或 uv
- 本地 X2Real 仿真资产包

### 1. 安装 ManaEnv

```bash
source ./install.sh
```

安装器会：

- 创建或激活 `manaenv` Conda 环境。
- 安装轻量的 `policy-space-protocol` 客户端。
- 按 [X2Real 依赖配置](.agents/skills/x2real-install/SKILL.md#runtime-dependency-profile) 构建可追溯的本地 Isaac Sim
  兼容 wheel，并要求 `pip check` 通过。

### 2. 准备本地资产

仓库只包含资产 manifest。

- **仿真资产：** 从 [X2Real 仿真资产](https://huggingface.co/datasets/x-square-robot/x2real-assets) 下载，将其中的 `assets/` 目录复制到本仓库。
- **数据集：** 查看单独的 [X2Real 数据集](https://huggingface.co/datasets/x-square-robot/X2Real-Dataset)；仓库不随附该数据集。

也可以让 `MANAENV_ASSET_ROOT` 指向本地 `assets/` 目录：

```bash
export MANAENV_ASSET_ROOT=/path/assets
```

`assets/manifest.json` 记录预期的资产版本、文件数、字节数和目录树摘要。

### 3. 启动 Policy Space 服务

在另一个 shell 中使用固定的 `3rd/policy_space` checkout：

```bash
cd 3rd/policy_space
python -m pip install --editable .
policy-space check policies/templates/bimanual_ee/deploy.yaml
policy-space serve policies/templates/bimanual_ee/deploy.yaml
```

仓库模板不需要 checkpoint，可用于接口验证。

### 4. 启动相连的评测

保持服务运行，在仓库根目录执行 headless ID 冒烟 case：

```bash
python manaenv/scripts/x2real_eval.py   --board artixon_arm6a_tabletop   --env artixon_arm6a_G_pick_fruits_into_basket_id_env   --num-episodes 1   --num-parallel-envs 1   --policy-config policy/x2real_ps@v1.0.0.yaml   --server-address 127.0.0.1   --server-port 8001   --enable_cameras
```

只有需要本地 Kit 窗口时才添加 `--viz kit`。

正式 benchmark 需要替换模板部署文件和 ManaEnv 策略配置，并保持策略名、schema、
机器人本体、host 和 port 一致。模板只返回零动作，只能验证接口。

<a id="evaluation-suite-zh"></a>

## 🧪 评测套件

| Board | 平台与配置 | ID case 数 |
| :-- | :-- | --: |
| `artixon_arm6a_tabletop` | ArtiXon Arm-6A 桌面任务，G/H 夹爪 | 38 |
| `artixon_arm6a_challenge` | ArtiXon Arm-6A 挑战任务，H 夹爪 | 10 |
| `quanta_x1_mobile` | Quanta X1 移动操作 | 3 |
| `arx_r5_tabletop` | ARX R5 桌面任务 | 12 |
| `franka_panda_tabletop` | Franka 桌面任务，Panda Hand | 10 |
| **合计** |  | **73** |

论文包含 44 个语义任务；评测器提供 73 个可执行 ID case。每个 case 默认运行 40 个
episode，并使用 4 个并行环境。

<a id="repository-structure-zh"></a>

## 🗂️ 仓库结构

```text
manaenv-pub/
├── install.py
├── install.sh
├── manaenv/
│   ├── scripts/x2real_eval.py
│   └── defination/
│       ├── x2real.yaml
│       └── x2real/task/
│           ├── boards.yaml
│           └── x2real_*_leaderboard.csv
├── assets/
├── 3rd/
│   ├── isaaclabarena/
│   │   └── submodules/IsaacLab/
│   │       └── IsaacSim/
│   └── policy_space/
└── outputs/x2real_eval/
```

安装器在 `3rd` 下准备依赖：

- Arena → IsaacLab → Isaac Sim
- Newton 通过 pip 使用 X-Square-Robot 的 `x2real` 引用
- Policy Space 跟踪 `main`

`outputs/x2real_eval/` 保存报告和视频。

<a id="outputs-and-scoring-zh"></a>

## 📦 输出与评分

报告默认写入 `outputs/x2real_eval/<board>/benchmark/`，视频位于：

```text
<save_path>/<environment>/videos/combined/episode_*.mp4
```

使用相同的筛选条件、episode 数量、seed 和保存路径，并添加 `--resume` 续跑中断任务：

```bash
python manaenv/scripts/x2real_eval.py   --board artixon_arm6a_tabletop   --env artixon_arm6a_G_pick_fruits_into_basket_id_env   --num-episodes 40   --seed 42   --save-path ./outputs/x2real_eval/resumable-run   --resume
```

报告路径记录在 `benchmark_report.json`。进程完成与任务 `success=true` 是两个不同信号。

<a id="manual-collection-zh"></a>

## 人工遥操作数采

在 ArtiXon Arm-6A 或 Quanta X1 的单个 ID 环境采集人工示范，无需策略服务。
先发现任务、检查配置和资产，再预览命令；移除 `--dry-run` 后启动：

```bash
python manaenv/scripts/x2real_collect.py --mode gendata --list
python manaenv/scripts/x2real_collect.py --mode gendata --check
python manaenv/scripts/x2real_collect.py --mode gendata --asset-check
python manaenv/scripts/x2real_collect.py --mode gendata \
  --env artixon_arm6a_G_pick_fruits_into_basket_id_env \
  --teleop keyboard --num-demos 10 --seed 42 \
  --save-path ./outputs/manual/fruits --dry-run
```

键盘模式需要 Kit 窗口。`B` 切换手臂／底盘，`W/S/A/D/Q/E` 移动，
`Z/X/T/G/C/V` 旋转，`K` 切换夹爪开合。`Enter` 保存示范，`R` 丢弃并重置。
可用 `--episode-length-s` 调整单次尝试的超时。
Kit 数采会打开相机预览，并在已适配的 G 夹爪腕部画面显示夹尖准星。
准星仅用于显示，不写入 HDF5 图像。设置 `MANAENV_EX001_6R_TIP_CROSSHAIR=0`
可关闭准星；未标定的 H 夹爪资产自动跳过。Kit 的 `Sim FPS` 窗口会显示实测数采
频率和单步耗时。`--viz none` 不打开预览或 FPS 窗口。

Vuer/WebXR 使用 `--teleop vuer`（或 `vr`），需在仿真环境安装可选依赖
`python -m pip install -e '.[teleop-vr]'`。主从臂使用
`--teleop masterslave --master-address <host> --master-port <port>`。
这两种方式支持 `--viz none`。设备设置和消息格式见
[gendata skill](.agents/skills/x2real-gendata/SKILL.md)。

`--num-demos` 只统计已经保存的成功示范。HDF5 包含动作、观测、状态、相机图像和元数据。
使用 `python manaenv/scripts/x2real_collect.py --inspect <saved-hdf5>` 检查输出。

<a id="release-scope-zh"></a>

## ⚖️ 范围

- 评测需要兼容的 Policy Space v2 端点和已注册的 ID case。
- 人工数采支持 ArtiXon Arm-6A、Quanta X1，提供键盘、Vuer/WebXR 和 WebSocket
  主从臂输入与本地 HDF5 导出，详见[操作说明](#manual-collection-zh)和[公开范围](OPEN_SOURCE_SCOPE.md)。
- 仿真资产与 [X2Real 数据集](https://huggingface.co/datasets/x-square-robot/X2Real-Dataset) 单独分发；仿真资产请从 [X2Real 仿真资产](https://huggingface.co/datasets/x-square-robot/x2real-assets) 下载。

<a id="citation-zh"></a>

## 📖 论文引用

研究中使用 X2Real 时，请引用我们的[论文](https://arxiv.org/abs/2609.27449)：

```bibtex
@misc{ruan2026x2realextensivesimulationbenchmark,
  title={X2Real: an eXtensive simulation benchmark for real-world generalist policies},
  author={Lian Ruan and Jade Yang and Sherphylan Gao and Felix Gao and
          Kyson Liang and Galen Liu and Ligo Wu and Lane Jin and Guu Gu and
          Bevan Xie and Cloud Yan and Zongzi Yuan and Kino Luo and Emma Chen and
          Shuwen Chen and Yang Ping and Miles Guo and Rain Sun and Kayden Zhang and
          Alex Du and Ruihai Wu and Liang Hao and Zhaoshuo Li and Roy Gan and
          Hao Wang and Qian Wang},
  year={2026},
  eprint={2609.27449},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2609.27449},
}
```

<a id="acknowledgements-zh"></a>

## 🙏 致谢

X2Real 建立在开源仿真与机器人基础设施之上，感谢：

- [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) 为项目提供了开源的 Isaac Lab 仿真评测基础设施。
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 为 Arena 提供基础框架。
- [Isaac Sim](https://github.com/isaac-sim/IsaacSim) 提供仿真运行时。
- [Newton](https://github.com/newton-physics/newton) 提供物理后端。

<a id="license-zh"></a>

## 📄 许可证

本仓库采用 [Apache License 2.0](LICENSE) 许可。第三方组件和单独分发的资产可能适用各自的许可证。
