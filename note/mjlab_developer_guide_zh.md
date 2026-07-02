# mjlab 开发者使用报告

> 基于仓库 README 与 `docs/` 文档整理。官方文档：https://mujocolab.github.io/mjlab/

---

## 1. mjlab 是什么

**mjlab** 是一个面向机器人强化学习的轻量级框架，核心定位是：

- 采用 **Isaac Lab 的 Manager-based API**（观测/奖励/终止/事件等模块化组合）
- 物理后端使用 **MuJoCo Warp**（GPU 加速的 MuJoCo）
- **PyTorch 原生**：观测、奖励、动作都是 GPU 上的 tensor，零拷贝共享内存
- **依赖少、启动快**：一条命令即可跑 demo

与同类框架对比：

| 框架 | 优势 | 适合场景 |
|------|------|----------|
| **mjlab** | 轻量、MuJoCo 原生、PyTorch、GPU 并行 | 想用 MuJoCo 做结构化 RL 环境 |
| **Isaac Lab** | 渲染、USD、Omniverse 生态 | 需要 Isaac Sim 能力 |
| **MuJoCo Playground** | 极简、易 hack | 单任务快速原型 |
| **Newton** | 多物理求解器、可微仿真 | 超出 MuJoCo 刚体范围 |

**设计哲学**：

1. 安装摩擦最小（`uvx --from mjlab --refresh demo`）
2. 物理透明可调试（直接访问 `MjModel`/`MjData`）
3. 深度集成 MuJoCo 生态（MJCF、Menagerie 等）

**范围**：刚体机器人学习；支持 depth/raycast 几何感知；高保真 RGB 渲染不是重点（常见做法：先用 privileged state 训练，再蒸馏到视觉策略）。

---

## 2. 系统架构（两层模型）

```
┌─────────────────────────────────────────────────────────┐
│  Manager Layer (MDP)                                     │
│  Observation / Action / Reward / Termination / Event /   │
│  Command / Curriculum / Metrics                          │
│  → ManagerBasedRlEnv                                     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Simulation Layer                                        │
│  Entity + Actuator + Sensor + Scene + Terrain            │
│  → MjSpec 组合 → MjModel → MuJoCo Warp (N parallel worlds)│
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
              RSL-RL (PPO 训练)
```

### 2.1 Simulation Layer（仿真层）

**Scene 流水线**：

1. 各 Entity 的 MJCF 通过 `MjSpec.from_file()` 加载
2. Python dataclass 可覆盖 actuator、碰撞、材质、传感器、初始状态
3. 组合成单个 `MjSpec` → 编译为 CPU 上的 `MjModel`
4. 上传到 GPU（MuJoCo Warp），单个 `MjData` 持有 **N 个并行 world**

**MuJoCo Warp 要点**：

- 模型参数默认所有 world 共享；DR 需要时可 per-world 扩展
- `step`/`forward`/`reset`/`sense` 被 **CUDA Graph** 捕获，消除 CPU dispatch 开销
- 每 episode 的 reset 和 DR 事件在 graph replay 之间以普通 Python 运行，不会破坏 capture

**四大组件**：Entity、Actuator、Sensor、Scene

### 2.2 Manager Layer（MDP 层）

环境由 **`ManagerBasedRlEnvCfg`** 这一个 dataclass 完全描述，包含 8 个 Manager：

| Manager | 作用 |
|---------|------|
| ObservationManager | 组装观测组（clip/noise/delay/history） |
| ActionManager | 将 policy 输出路由到 actuator |
| RewardManager | 加权求和 reward terms |
| TerminationManager | 终止/截断条件 |
| EventManager | startup/reset/interval/step 事件 |
| CommandManager | 速度目标、参考动作等 |
| CurriculumManager | 按性能调整难度 |
| MetricsManager | 诊断指标（不影响优化） |

### 2.3 环境 Step 顺序

```
action_manager.process_action(action)
for _ in range(decimation):
    action_manager.apply_action()
    sim.step()
    scene.update()
termination_manager.compute()
reward_manager.compute()
metrics_manager.compute()
[reset terminated envs]
sim.forward()
command_manager.compute()
event_manager.apply(mode="interval")
sim.sense()
observation_manager.compute()
```

**时间参数**：

- `physics_dt = sim.mujoco.timestep`（默认 0.002s = 500Hz）
- `step_dt = physics_dt × decimation`
- `scale_rewards_by_dt=True`（默认）使 episodic reward 与仿真频率无关

---

## 3. 安装与环境要求

| 用途 | 平台 | 要求 |
|------|------|------|
| **训练** | Linux + NVIDIA GPU | CUDA 12.4+ 推荐 |
| **评估** | Linux / macOS / WSL | macOS 仅 CPU，慢 |
| **Python** | 3.10–3.13 | 用 `uv`，不用裸 `python` |

### 安装方式速查

```bash
# 作为依赖
uv add mjlab && uv run demo

# 开发 mjlab 本身
git clone ... && cd mjlab && uv sync && uv run demo

# 零安装
uvx --from mjlab --refresh demo

# Docker
docker run --rm --runtime=nvidia --gpus all ghcr.io/mujocolab/mjlab uv run demo
```

---

## 4. CLI 命令

| 命令 | 作用 |
|------|------|
| `uv run demo` | 快速演示 |
| `uv run train <TASK_ID>` | 训练 |
| `uv run play <TASK_ID>` | 回放/调试 |
| `uv run list-envs` | 列出任务 |
| `uv run export-scene` | 导出 MJCF |
| `uv run viz-nan` | NaN dump 可视化 |

---

## 5. 内置任务

| Task ID | 说明 |
|---------|------|
| `Mjlab-Cartpole-Balance` / `Swingup` | 入门 |
| `Mjlab-Velocity-Flat/Rough-Unitree-G1` | G1 速度跟踪 |
| `Mjlab-Velocity-Flat/Rough-Unitree-Go1` | Go1 四足 |
| `Mjlab-Tracking-Flat-Unitree-G1` | 动作模仿 |
| `Mjlab-Lift-Cube-Yam` 等 |  manipulation |

---

## 6. 构建自定义环境（Cartpole 模式）

1. 写 MJCF
2. `EntityCfg(spec_fn=..., articulation=..., init_state=...)`
3. 配置 observations / actions / rewards / terminations / events
4. 组装 `ManagerBasedRlEnvCfg`
5. `register_mjlab_task(...)` 注册
6. `uv run train Mjlab-Your-Task ...`

推荐 **工厂函数 + dataclasses.replace** 模式（见 velocity 任务）。

---

## 7. 训练示例

```bash
# 速度跟踪
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096

# 多 GPU
uv run train Mjlab-Velocity-Flat-Unitree-G1 --gpu-ids "[0, 1]" --env.scene.num-envs 4096

# 动作模仿（需 W&B registry）
uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --registry-name your-org/motions/motion-name --env.scene.num-envs 4096

# 评估
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path org/project/run-id

# Sanity check
uv run play Mjlab-Your-Task --agent zero
```

训练产物：`logs/rsl_rl/{experiment_name}/{timestamp}/model_*.pt`

---

## 8. 核心概念摘要

- **Entity**：MJCF + Python 配置；fixed-base 需 reset event 才能 spread 到 grid
- **Actuator**：Built-in（隐式积分，稳定）vs Explicit（自定义控制律）vs XmlActuator
- **Scene**：组合 entity，prefix 命名（`robot/joint0`）
- **Observations**：compute → noise → clip → scale → delay → history
- **DR**：`mjlab.envs.mdp.dr` + EventTermCfg；`pseudo_inertia` 做物理一致的质量/惯量随机化
- **Terrain**：plane 或 generator + curriculum

---

## 9. 开发 mjlab 本身的工作流

```bash
make format      # ruff
make type        # ty + pyright
make check       # format + type
make test-fast   # 快速测试
make test        # 全量
make docs        # 文档
```

**注意**：这是改 mjlab 源码、提 PR 用的。日常 RL 训练只需 `uv run train/play`，不需要 `make`。

---

## 10. 源码目录

```
src/mjlab/
├── envs/          # ManagerBasedRlEnv, mdp/
├── managers/      # 8 个 Manager
├── entity/        # Entity, EntityCfg
├── actuator/      # 执行器
├── sensor/        # 传感器
├── scene/         # Scene 组合
├── sim/           # MuJoCo Warp 封装
├── terrains/      # 程序化地形
├── tasks/         # 内置任务 + registry
├── asset_zoo/     # Go1, G1, YAM MJCF
├── rl/            # RSL-RL 配置
├── viewer/        # native / viser
└── scripts/       # train, play, demo
```

---

## 附录 A：概念问答（开发者常见问题）

### A.1 什么是 MuJoCo Warp？

**MuJoCo Warp**（`mujoco_warp`）是 Google DeepMind 基于 **NVIDIA Warp** 做的 MuJoCo **GPU 加速后端**。

- 保留 MuJoCo 的 `MjModel` / `MjData` 概念，但 `MjData` 多一维 **world**：一次 step 可并行推进 N 个独立仿真实例（vectorized env）。
- mjlab 用它做大规模并行 RL（4096+ envs），并用 CUDA Graph 减少 CPU 调度开销。
- 与 CPU 版 MuJoCo 的关系：同一套 MJCF 建模，物理语义一致，但实现路径在 GPU 上；目前仍在 beta， determinism 等特性在完善中。

**mjlab 不是简单「包一层接口」**：仿真层确实封装了 Warp + PyTorch tensor 视图；但更上层还有 Scene 组合、Manager MDP、Task Registry、RSL-RL 集成等，是完整 RL 框架而非 thin wrapper。

### A.2 什么是 Python dataclass？

`@dataclass` 是 Python 3.7+ 的装饰器，用**声明式字段**自动生成 `__init__`、`__repr__` 等样板代码。

mjlab 里 `ManagerBasedRlEnvCfg`、`EntityCfg`、`RewardTermCfg` 等都是 dataclass：配置即数据，扁平可读，拼错字段名会立刻报错（对比 Isaac Lab 嵌套 `@configclass` + `__post_init__`）。

```python
from dataclasses import dataclass, field

@dataclass
class MyEnvCfg(ManagerBasedRlEnvCfg):
    decimation: int = 4
    scene: SceneCfg = ...
    rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
```

### A.3 regex 是什么？

**regex = regular expression（正则表达式）**，一种用模式字符串匹配文本/名称的语法。

mjlab 里大量用于**按名字选 joint/body/geom/actuator**，例如：

- `".*_hip_.*"` 匹配所有含 `_hip_` 的关节名
- `("slider",)` 精确匹配

在 **Manager 初始化时**解析成整数 index，runtime 不再做 regex，无性能开销。相关配置：`SceneEntityCfg(..., joint_names=(".*",))`、`ActuatorCfg.target_names_expr`。

### A.4 为什么推荐 Built-in（隐式积分）执行器？

这里的「隐式/显式」指的是 **数值积分如何处理速度相关力（尤其阻尼）**，不是「隐式=不真实」。

| | Built-in（MuJoCo 原生 actuator） | Explicit（Python 算力矩 + motor 透传） |
|--|--|--|
| 积分 | 阻尼项可被 `implicitfast` **隐式**处理 | 力矩在 Python 算完再注入，积分器**看不到**阻尼导数 |
| 稳定性 | 大 timestep、高增益更稳 | 同样增益下更容易数值发散 |
| 适用 | 常规定位 PD、velocity、motor | 自定义控制律、学习到的 actuator 模型 |

**贴近现实**靠的是：电机参数（armature、effort limit）、`DcMotorActuator` 力矩-速度曲线、`delay_*` 命令延迟、`dr.encoder_bias` 等——这些 explicit 类型也能做。Built-in 推荐的首要原因是 **仿真稳定、训练少 NaN**，不是因为它更「假」。

大 timestep 下 explicit PD 和 built-in 在线性区接近；RL 常用 decimation 较大时，built-in 更省心。

### A.5 文档里有 sim2real 教程吗？

**没有独立的 sim2real 专章**，但文档 scattered 地提供了 sim2real **常用工具**：

- **Domain Randomization**（`dr.*`）：摩擦、质量/惯量、COM、PD 增益、encoder bias 等
- **Observation noise / delay**：传感器延迟与噪声
- **Actuator delay**：命令链路延迟
- **Asymmetric actor-critic**：actor 只用 onboard 可观测量，critic 用 privileged state
- **`joint_pos_rel(biased=True)`** + `dr.encoder_bias`：编码器偏置

整体思路是 **DR + 观测/执行器建模 → 部署时再处理 gap**，不是端到端「从仿真到实机步骤清单」。要 sim2real 还需自己：实机接口、安全层、system ID、可能 distillation 等。

### A.6 W&B 是什么？和 TensorBoard 一样吗？

**W&B = Weights & Biases**（https://wandb.ai），云端/本地的 **实验跟踪与协作平台**。

与 TensorBoard 的对比：

| | TensorBoard | W&B |
|--|-------------|-----|
| 定位 | 本地 scalar/曲线/图像 | 实验管理 + 指标 + artifact + sweep |
| 部署 | 本地 `tensorboard --logdir` | 默认上传云端 dashboard |
| mjlab 用法 | RSL-RL 也写 tb | checkpoint、motion registry、sweep 与 mjlab 深度集成 |

mjlab 训练默认 log 到 W&B；也可关 upload。motion imitation 的 **reference motion 存在 W&B registry**，这是 TensorBoard 做不到的。

### A.7 云端训练 vs 实验室服务器工作流

**文档里的「云端训练」**指 SkyPilot + **Lambda Cloud 等公有云 GPU 租赁**：按需开机器、跑完自动关、按小时计费，适合没固定 GPU 或要 burst 算力。

**你的工作流**（本地开发 → git push → 实验室服务器 pull → 跑训练）是 **自建集群/固定服务器** 模式，完全合理且很多实验室主流做法。

| | 实验室服务器 | SkyPilot 云端 |
|--|-------------|---------------|
| 成本 | 摊销/免费额度 | 按 GPU 小时 |
| 环境 | 自己维护 CUDA/依赖 | YAML/Docker 一键 |
| 数据/代码 | git/rsync 熟悉 | SkyPilot rsync + 自动 provision |
| 适合 | 有稳定 GPU、长期项目 | 没 GPU、临时大规模 sweep |

**结论**：你有实验室服务器，**继续 git → 服务器跑** 通常更好；云端文档是可选方案，不是必须。mjlab 在两种环境下都是 `uv run train ...`，差别在机器从哪来。

### A.8 开发工作流是开发 mjlab 还是训练？

**对。** `make format/type/test/check` 是给 **改 mjlab 源码、提 PR** 的：

- 改 Python 包、跑 pytest、过 ruff/pyright
- 改文档 `make docs`

**RL 训练/调参**：

```bash
uv run train ...
uv run play ...
```

纯 Python CLI + tyro，**不需要 make**。只有当你 fork mjlab 改框架本身时才需要完整 dev workflow。

---

## 附录 B：数值积分与执行器（详解）

### B.1 从连续动力学到离散仿真

机器人关节动力学（简化）：

\[
M(q)\ddot{q} + C(q,\dot{q}) + g(q) = \tau
\]

MuJoCo 每个 **physics step**（`timestep`，如 0.002s）做一件事：**已知当前** \(q, \dot{q}\) **和控制输入**，算出下一步的 \(q, \dot{q}\)。这就是**数值积分**（离散化）。

RL 里还有 **decimation**：policy 每 N 个 physics step 才输出一次新 action，中间 N-1 步沿用同一目标。

### B.2 PD 控制里有什么「速度相关力」

位置 PD（理想情况）：

\[
\tau = k_p (q_{\text{target}} - q) + k_d (\dot{q}_{\text{target}} - \dot{q})
\]

**阻尼项** \(-k_d \dot{q}\) 与速度成正比。它出现在运动方程右侧，且**随 \(\dot{q}\) 变化**——积分器必须正确处理这类力，否则大 timestep 下会振荡或发散（类似欧拉法解 stiff 系统不稳定）。

### B.3 显式 vs 隐式积分（核心区别）

以 \(\dot{x} = f(x)\) 为例：

| 方式 | 公式（概念） | 特点 |
|------|-------------|------|
| **显式** | \(x_{k+1} = x_k + \Delta t \cdot f(x_k)\) | 用**当前步**的力推进；阻尼大、dt 大时容易不稳定 |
| **隐式** | 求解 \(x_{k+1} = x_k + \Delta t \cdot f(x_{k+1})\) | 力与**下一步状态**耦合；阻尼项可被稳定处理，允许更大 dt |

MuJoCo 的 `implicitfast` 会对**已知的**速度相关力（含 actuator 阻尼）做隐式处理。

### B.4 Built-in vs Explicit 执行器在 mjlab 里差在哪

**Built-in**（`BuiltinPositionActuator` 等）：

- 在 MjSpec 里创建 MuJoCo 原生 `<position>` / `<velocity>` / `<motor>` 等
- **控制律在仿真器内部**，积分器知道 \(\tau\) 如何依赖 \(\dot{q}\)
- 阻尼走 **implicitfast 的隐式路径** → 高 \(k_p, k_d\) 或大 `timestep` 仍较稳

**Explicit**（`IdealPdActuator` 等）：

- **Python 先算** \(\tau = k_p e + k_d \dot{e}\)，再通过 `<motor>` **透传**力矩
- 对积分器来说，\(\tau\) 是「外部施加的力」，**结构未知**，无法对阻尼项做隐式处理
- 同样增益下，数值上更「硬」，容易 NaN/爆炸

**重要澄清**：Explicit **可以**更贴近某些真实电机模型（如 `DcMotorActuator` 力矩-速度饱和、`LearnedMlpActuator`）。Built-in 推荐是因为 **RL 训练要跑百万步、大 batch、较大 dt**——**数值稳定优先**；不是 built-in 更「假」。

小 timestep、低增益时两者接近；mjlab 默认 `timestep=0.002~0.005`、`decimation=4`，built-in 更省心。

### B.5 integrator：`euler` vs `implicitfast`

mjlab 默认 **`implicitfast`**（MuJoCo 推荐）。

- 关节上的 passive damping：euler 可隐式处理
- **Actuator 上的 damping**：euler 往往**显式** → 不稳定
- `implicitfast`：actuator 的 P 和 D 项都尽量隐式处理

文档原话：mjlab 把 damping 放在 actuator 而非 joint 上，所以 integrator 选型对执行器稳定性影响很大。

### B.6 和「贴近现实」的关系

| 建模手段 | Built-in 能否做 | Explicit 典型用途 |
|---------|----------------|------------------|
| 力矩/速度限制 | `effort_limit`、BuiltinDcMotor | `DcMotorActuator` 力矩-速度曲线 |
| 转子惯量 | `armature` | 同左 |
| 命令延迟 | `delay_min_lag/max_lag` | 同左 |
| 学习到的 actuator | — | `LearnedMlpActuator` |
| 自定义非线性控制律 | 受限 | Python 里任意算 \(\tau\) |

**Sim2real  gap** 更多来自 DR、传感器噪声、延迟、encoder bias，而不是 built-in vs explicit 二选一。

### B.7 直觉小结

```
Built-in PD  →  仿真器「认识」这是 PD，阻尼可隐式积分  →  稳
Explicit PD  →  仿真器只看到「一坨力矩」            →  同样增益下更脆
```

需要 custom actuator 动力学时用 Explicit；常规 locomotion 用 Built-in 是**工程默认**，不是物理真理。

---

## 附录 C：如何监视训练过程

### C.1 默认：Weights & Biases（W&B）

`RslRlBaseRunnerCfg.logger` 默认 `"wandb"`。训练启动后：

1. 浏览器打开 https://wandb.ai 对应 project（默认 `mjlab`，可用 `--agent.wandb-project` 改）
2. 看 reward 曲线、PPO loss、学习率、自定义 metrics 等

首次使用需登录：

```bash
uv tool install wandb   # 或 pip install wandb
wandb login
```

常用 CLI：

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.experiment-name g1_velocity \
  --agent.run-name trial_01 \
  --agent.wandb-tags "[baseline, flat]"
```

关闭 W&B 模型上传（仍 log 指标）：

```bash
--agent.upload-model False
```

### C.2 备选：TensorBoard

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --agent.logger tensorboard \
  --env.scene.num-envs 4096
```

另开终端：

```bash
tensorboard --logdir logs/rsl_rl/g1_velocity
# 浏览器打开 http://localhost:6006
```

RSL-RL 会把 scalar 写到 `{log_dir}/` 下 TensorBoard event 文件。Tracking 任务若用 tensorboard，不会走 W&B artifact/registry 那套。

### C.3 本地日志目录结构

```
logs/rsl_rl/{experiment_name}/{timestamp}_{run_name}/
  model_{iter}.pt              # checkpoint
  params/env.yaml              # 完整环境配置
  params/agent.yaml            # PPO/网络配置
  videos/train/                # 若 --video True
  torchrunx/                   # 多 GPU 时子进程日志
```

`--log-root` 可改根目录（默认 `logs/rsl_rl`）。

### C.4 训练时会 log 哪些量

**RSL-RL / PPO**（两种 logger 都有）：policy loss、value loss、entropy、learning rate、approx KL、FPS 等。

**mjlab Manager 层**（episode 结束时写入 extras，再进 logger）：

| 前缀 | 含义 |
|------|------|
| `Episode_Reward/<term>` | 各 reward term 的 episode 平均（已按 dt 缩放） |
| `Episode_Metrics/<term>` | 诊断量（跟踪误差等，不参与优化） |
| `Episode_Termination/<term>` | 各终止原因触发比例 |
| `Curriculum/<term>` | curriculum 项返回值 |

NaN 终止若配置了 `nan_detection`，可看 `Episode_Termination/nan_term`。

### C.5 训练时录视频

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --video True \
  --video-interval 2000 \
  --video-length 200
```

视频在 `logs/.../videos/train/`；W&B 模式下也可能出现在 run 的 Media 面板（取决于 RSL-RL 上传行为）。

### C.6 训练中 vs 训练后监视

| 阶段 | 手段 |
|------|------|
| **训练中** | W&B / TensorBoard 实时曲线；`--video` 抽查 rollout |
| **训练后** | `uv run play ... --wandb-run-path` 或 `--checkpoint-file`；native/viser viewer |
| **调试数值** | `--enable-nan-guard True` → `uv run viz-nan ...` |
| **看 reward 分项** | play 时 native viewer 按 `P` 键 |

### C.7 实验室服务器上的推荐做法

有固定 GPU 服务器时：

1. **W&B online**：服务器能访问外网即可，本地/手机都能看曲线（最省事）
2. **TensorBoard + SSH 隧道**：`ssh -L 6006:localhost:6006 user@server`，本地浏览器看
3. **仅本地文件**：看 `logs/rsl_rl/.../`，用 tensorboard 或事后 play

W&B 与 TensorBoard **二选一**（`--agent.logger`），不是同时默认开启；checkpoint 始终落本地磁盘。

---

## 引用

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```
