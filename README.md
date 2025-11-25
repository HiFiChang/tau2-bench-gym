# τ²-bench Gym(Gymnasium) 环境适配

## 概述

本repo在原始 τ²-bench 基础上新增了 **Gym 兼容的强化学习环境**，使得该基准测试框架可以直接用于强化学习研究和训练。核心实现包括：

- **`src/tau2/environment/gym_env.py`**: Gym 环境封装类
- **`run_telecom_tasks.py`**: 示例运行脚本

## 设计思路

### 核心理念

τ²-bench 原本是一个用于评估对话型客服代理的基准测试框架，其核心是 **Orchestrator** 系统，它协调三个角色之间的交互：
- **Agent（代理）**: 客服代理，需要根据策略帮助用户
- **User（用户）**: 通过 LLM 驱动的用户模拟器，向代理提出问题
- **Environment（环境）**: 执行工具调用并返回结果

为了将这个框架适配为标准的 Gym 环境，需要明确**从谁的视角**来定义强化学习问题。

### Agent 为学习主体

**Agent作为强化学习的主体**，因为目标是训练更好的assistant，Agent 需要学习如何与用户交互、使用工具解决问题。因此，在 Gym 环境中：
- **Agent = RL Agent（学习主体）**
- **User + Environment = Gym Environment（环境）**

### Gym 接口映射

基于上述设计理念，我们将 τ²-bench 映射到标准的 Gym 接口：

| Gym 概念 | τ²-bench 对应 | 说明 |
|---------|--------------|------|
| Action | `AssistantMessage` | Agent 发送的消息（文本回复或工具调用） |
| Observation | `Message` | Agent 接收到的消息（用户消息或工具返回） |
| Reward | τ² 评估器计算的分数 | 回合结束时计算（0-1 分） |
| Episode | 一个完整的任务对话 | 从问候开始到问题解决或达到步数上限 |
| Done | 终止条件 | Agent/User 停止、达到最大步数或错误数 |
## 实现逻辑

### 1. 环境初始化 (`Tau2GymEnv.__init__`)

初始化环境，加载指定领域的任务。

### 2. 回合重置 (`reset()`)

每次调用 `reset()` 开始新回合：

1. **选择任务**: 从任务列表中选择一个（循环或指定）
2. **创建组件**: 
   - 实例化 Environment（领域环境，提供工具）
   - 实例化 UserSimulator（LLM 驱动的用户）
   - 创建 DummyAgent（占位符，实际动作由外部策略提供）
3. **初始化 Orchestrator**: 创建协调器管理三方交互
4. **执行初始交互**: 
   - Agent 发送问候："Hi! How can I help you today?"
   - User 回复描述问题
5. **返回初始观察**: 返回用户的首条消息作为 observation

### 3. 执行步骤 (`step(action)`)

这是 Gym 环境的核心逻辑，每次调用代表 Agent 执行一个动作：

**输入**: 
- `action`: 一个 `AssistantMessage` 对象（包含文本或工具调用）

**处理流程**:
1. **验证动作**: 检查动作格式是否合法
2. **注入 Orchestrator**: 将动作设置为 Orchestrator 的当前消息
3. **确定流向**: 
   - 如果是工具调用 → `to_role = Environment`
   - 如果是文本消息 → `to_role = User`
4. **内部循环执行**: 持续调用 `orchestrator.step()`，直到控制权返回 Agent
   - User 接收消息 → 生成回复 → 发给 Agent
   - Environment 执行工具 → 返回结果 → 发给 Agent
   - **可能涉及多轮内部交互（ User 使用工具）**
5. **检查终止条件**:
   - Agent/User 发送停止信号
   - 达到最大步数
   - 工具错误次数过多
6. **计算奖励**: 只在回合结束时计算最终奖励（使用 τ² 评估器）

**输出**:
- `observation`: 下一条发给 Agent 的消息
- `reward`: 回合中为 0.0，结束时为最终分数（0-1）
- `terminated`: 自然结束（Agent/User 停止）
- `truncated`: 达到限制（步数/错误数）
- `info`: 包含步骤计数、终止原因、奖励分解等元数据

### 4. 关键设计点

#### 内部步骤隐藏
一次 `gym.step(action)` 可能对应多次内部交互：
```
Agent 动作 → User 思考（可能使用工具）→ Environment 响应 → User 生成回复 → 返回 Agent
```
所有这些中间步骤对 RL Agent **不可见**，只看到最终返回的观察。

#### 奖励延迟
强化学习中的 sparse reward：
- 回合进行中：`reward = 0.0`
- 回合结束时：`reward = evaluate_simulation(...)` （0-1 分）

#### 消息历史管理
Orchestrator 维护完整的对话历史（`trajectory`），策略可以访问这个历史来生成下一个动作。

## 使用方法

首先**需要按照τ²-bench的安装说明完成环境配置**。请参看[Installation](#installation)。

我们提供了 `run_telecom_tasks.py` 作为脚本：

**运行所有 telecom 任务**:
```bash
python run_telecom_tasks.py --domain telecom --num-trials 1
```

**运行特定任务**:
```bash
python run_telecom_tasks.py --task-ids '[mobile_data_issue]data_mode_off|data_usage_exceeded[PERSONA:None]'
```

**指定 LLM 模型**:
```bash
python run_telecom_tasks.py \
  --agent-llm gpt-4o \
  --user-llm gpt-4.1 \
  --num-trials 2
```

**运行前k个任务**:
```bash
python run_telecom_tasks.py --num-tasks 5 --num-trials 1
```

### 输出说明

脚本会生成两类输出：

1. **汇总结果** (`results/telecom_run_TIMESTAMP.json`):
   - 平均奖励
   - 成功率
   - 每个任务的详细结果

2. **轨迹文件** (`results/telecom_trajectories/TASK_ID_trial_TIMESTAMP.json`):
   - **Agent 视角**: 完整的 messages、tools、system_prompt（可用于复现）
   - **User 视角**: User Simulator 的内部状态
   - **完整对话**: 第三方观察视角的对话记录
   - **步骤级轨迹**: 每步的 observation-action-reward
   - **评估结果**: 最终奖励和奖励分解

## 其他细节

### 消息类型

- **`SystemMessage`**: 系统提示（策略、指令）
- **`UserMessage`**: 用户发送的消息
- **`AssistantMessage`**: Agent 发送的消息（文本或工具调用）
- **`ToolMessage`**: 工具执行结果
- **`MultiToolMessage`**: 多个工具结果的集合

### 终止原因

- **`AGENT_STOP`**: Agent 主动结束对话
- **`USER_STOP`**: User 主动结束对话  
- **`MAX_STEPS`**: 达到最大步数
- **`TOO_MANY_ERRORS`**: 工具错误次数过多

### 评估指标

调用 τ² 原有评估器。

## 与原仓库的关系

本适配**完全兼容**原始 τ²-bench 框架：
- 复用所有领域定义、任务、评估器
- 保持原有的 Orchestrator 逻辑
- **仅进行了增量修改**，在外层添加 Gym 接口包装
- 不影响原有的 CLI 和评估流程

---

**以下是原仓库的 README**

---

# $\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment

[![python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![arXiv](http://img.shields.io/badge/cs.AI-arXiv%3A2506.07982-B31B1B.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2506.07982)
[![blog](https://img.shields.io/badge/blog-tau2--bench-green)](https://sierra.ai/blog/benchmarking-agents-in-collaborative-real-world-scenarios)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/sierra.svg?style=social&label=Follow%20%40SierraPlatform)](https://x.com/SierraPlatform/status/1932464265207889974)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/posts/sierra_last-year-we-introduced-%F0%9D%9C%8F-bench-a-benchmark-activity-7338229693898231809-F8L4?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAdc8goBmhEsiEo1_t_XSJbAnY4_zMfAWcE)
[![Leaderboard](https://img.shields.io/badge/🏆_Live_Leaderboard-taubench.com-brightgreen?style=flat)](https://taubench.com)

<div align="center">
<img src="figs/overview.png" width="95%" alt="System Overview"><br>
<em>Figure 1: τ²-bench allows users to interact with the agent and the environment</em>
</div>

<div align="center">
<img src="figs/traj.png" width="95%" alt="Trajectory"><br>
<em>Figure 2: Trajectory of a conversation between an agent and a user</em>
</div>

## 🆕 What's New

### 🤖 Reinforcement Learning Support (New!)
τ²-bench now supports RL training with a Gymnasium-compatible interface:

- **🏋️ Train RL Agents**: Use the gym interface to train agents with popular RL frameworks. 
- **🎮 Play as Agent or User**: Interactive mode lets you control either the agent or the user in conversations
- **📊 Train/Test Splits**: To help support experiments around training Agents and evaluating them, all domains include standardized task splits for proper train/test evaluation.

> **⚠️ IMPORTANT FOR BACKWARD COMPATIBILITY**: If you are just evaluating an agent (not training), you **MUST** use the `base` task split to evaluate on the complete task set that matches the original τ²-bench structure. This ensures your results are comparable to previous evaluations and maintains consistency with the established benchmark. (If you don't specify a task split, it will default to `base`.)
- **🔧 Gymnasium Compatible**: Standard gym interface works with existing RL tools and libraries

[**→ See Gym Documentation**](src/tau2/gym/README.md) | [**→ Try CLI Play Mode**](#interactive-play-mode)

### 🏆 Live Leaderboard (v0.2.0)
The τ²-bench leaderboard is now live at **[taubench.com](https://taubench.com)**! 

- **📊 Interactive Rankings**: Compare model performance across all domains
- **📱 Mobile-Friendly**: View results on any device  
- **🔍 Detailed Analysis**: Explore trajectories and conversation flows
- **📥 Easy Submission**: Submit your results directly through the interface

[**→ Visit the Leaderboard**](https://taubench.com) | [**→ Submit Your Results**](#leaderboard-submission)

## Overview

$\tau^2$-bench implements a simulation framework for evaluating customer service agents across various domains.

**$\tau^2$-bench is the new iteration of the original $\tau$-bench**, featuring code fixes and an additional telecom domain.

Each domain specifies:
- a policy that the agent must follow
- a set of tools that the agent can use
- a set of tasks to evaluate the agent's performance
- Optionally: A set of tools that the user simulator can use

Domains are:
- `mock`
- `airline`
- `retail`
- `telecom`

All the information that an agent developer needs to build an agent for a domain can be accessed through the domain's API docs. See [View domain documentation](#view-domain-documentation) for more details.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
```

2. Create a new environment (optional)

$\tau^2$-bench requires Python 3.10 or higher. You may create and activate a new environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install tau2

```bash
pip install -e .
```

This will enable you to run the `tau2` command.

**Note:** If you use `pip install .` (without `-e`), you'll need to set the `TAU2_DATA_DIR` environment variable to point to your data directory:

```bash
export TAU2_DATA_DIR=/path/to/your/tau2-bench/data
```

**Check your data directory setup:**

After installation, you can verify that your data directory is correctly configured by running:

```bash
tau2 check-data
```

This command will check if the data directory exists and print instructions if it is missing.

To remove all the generated files and the virtual environment, run:
```bash
make clean
```

## Quick Start

### Setup LLM API keys

We use [LiteLLM](https://github.com/BerriAI/litellm) to manage LLM APIs, so you can use any LLM provider supported by LiteLLM.

To provide your API keys, copy `.env.example` as `.env` and edit it to include your API keys.

### Run agent evaluation

To run a test evaluation on only 5 tasks with 1 trial per task, run:

```bash
tau2 run \ 
--domain airline \
--agent-llm gpt-4.1 \
--user-llm gpt-4.1 \
--num-trials 1 \
--num-tasks 5
```

Results will be saved in `data/tau2/simulations/`.

> **💡 Tip**: For full agent evaluation that matches the original τ²-bench methodology, remove `--num-tasks` and use `--task-split base` to evaluate on the complete task set.

## Command Line Interface

The `tau2` command provides a unified interface for all functionality:

### Running Benchmark 
```bash
tau2 run \
  --domain <domain> \
  --agent-llm <llm_name> \
  --user-llm <llm_name> \
  --num-trials <trial_count> \
  --task-ids <task_ids> \
  --max-concurrency <concurrent_sims> \
  ...
```

### Interactive Play Mode
```bash
tau2 play
```
Experience τ²-bench from either perspective! The play mode allows you to:
- **Play as Agent**: Manually control the agent's responses and tool calls
- **Play as User**: Control the user while an LLM agent handles requests (available in domains with user tools like telecom)
- **Understand tasks** by walking through scenarios step-by-step
- **Test strategies** before implementing them in code
- **Choose task splits** to practice on training data or test on held-out tasks

This is perfect for:
- Getting familiar with domain policies and tools from both perspectives
- Debugging task scenarios and conversation flows
- Developing intuition for agent strategies
- Testing user behavior and agent responses
- Training yourself before training your model!

See the [Gym Documentation](src/tau2/gym/README.md) for more details on using the gymnasium interface programmatically, including the `AgentGymEnv` (play as agent) and `UserGymEnv` (play as user).

### Viewing Results
```bash
tau2 view
```
This tool allows you to:
- Browse simulation files (in `data/tau2/simulations/`)
- View agent performance metrics
- View a particular simulation
- View task details

### View domain documentation
```bash
tau2 domain <domain>
```
Visit http://127.0.0.1:8004/redoc to see the domain policy and API documentation.

![domain_viewer1](figs/domain_viewer.png)

### Check data configuration
```bash
tau2 check-data
```
This command checks if your data directory is properly configured and all required files are present.

## Leaderboard Submission

To submit your agent results to the τ²-bench leaderboard, you need to prepare a valid submission package that meets specific requirements.

### Requirements for Valid Submissions

Your trajectory runs must follow these constraints:

1. **Complete domain coverage**: Include results for all three domains:
   - `retail`
   - `airline` 
   - `telecom`

2. **Consistent model configuration**: All trajectory files must use:
   - The same agent LLM with identical arguments across all domains
   - The same user simulator LLM with identical arguments across all domains

3. **One result per domain**: Each domain should appear exactly once in your submission

4. **All tasks completed**: Run evaluation on all tasks within each domain (don't use `--task-ids` or `--num-tasks` filters)

> **📝 Note**: For consistency with the original τ²-bench evaluation methodology, use the `base` task split when evaluating your agent to ensure you're testing on the complete, standard task set.

### Preparing Your Submission

#### Step 1: Run Evaluations
First, run your agent evaluation on all domains with consistent settings:

```bash
# Example: Run complete evaluation for all domains
tau2 run --domain retail --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_retail
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_airline  
tau2 run --domain telecom --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_telecom
```

**Important**: Use identical `--agent-llm`, `--user-llm`, and their arguments across all runs.

#### Step 2: Prepare Submission Package
Use the submission preparation tool to create your leaderboard submission:

```bash
tau2 submit prepare data/tau2/simulations/my_model_*.json --output ./my_submission
```

This command will:
- Verify all trajectory files are valid
- Check that submission requirements are met
- Compute performance metrics (Pass^k rates)
- Prompt for required metadata (model name, organization, contact email)
- Create a structured submission directory with:
  - `submission.json`: Metadata and metrics
  - `trajectories/`: Your trajectory files

#### Step 3: Validate Your Submission
Before submitting, validate your submission package:

```bash
tau2 submit validate ./my_submission
```

This will verify:
- All required files are present
- Trajectory files are valid
- Domain coverage is complete
- Model configurations are consistent

### Additional Options

#### Skip Verification (if needed)
```bash
tau2 submit prepare data/tau2/simulations/my_model_*.json --output ./my_submission --no-verify
```

#### Verify Individual Trajectory Files
```bash
tau2 submit verify-trajs data/tau2/simulations/my_model_*.json
```

### Submitting to the Leaderboard

Once your submission package is prepared and validated:

1. Review the generated `submission.json` file
2. Follow the submission guidelines in [web/leaderboard/public/submissions/README.md](web/leaderboard/public/submissions/README.md) to create a Pull Request
3. Keep your `trajectories/` directory for reference

The leaderboard will display your model's Pass^k success rates (k=1,2,3,4) across all domains.

## Experiments

### Experimental Code Directory

The `@experiments/` directory contains experimental features and research code that extends beyond the core tau2 benchmark. This directory is designed for community contributions of innovative approaches, prototypes, and new features that are not part of the core evaluation framework.

- **Purpose**: Research code and experimental features
- **Location**: `src/experiments/`
- **Usage**: Each experimental component has its own README with documentation
- **Status**: Experimental code is provided as-is and may not be fully tested or supported

For more details, see the [experiments README](src/experiments/README.md).

### Running Ablation Studies (No User, or Agent with Oracle Plan)
`telecom` domain enables running ablation studies.

1. Running an LLM in `no-user` mode. In this mode, the LLM is given all the tools and the information upfront.
Just choose `llm_agent_solo` as the agent and `dummy_user` as the user.

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_solo \
  --agent-llm gpt-4.1 \
  --user dummy_user \
  ...
```

2. Running an LLM in `oracle-plan` mode. In this mode, the LLM is given an oracle plan ahead of time alleviating the need for action planning.
Just choose `llm_agent_gt` as the agent.

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_gt \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  ...
```

### Running Telecom Domain with Workflow Policy
To test the impact of policy format, we provide an additional "workflow" policy for the telecom domain.
To run using this policy, use the `telecom-workflow` domain.

```bash
tau2 run \
  --domain telecom-workflow \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  ...
```

## Domains

For all the details see the domains [README](src/tau2/domains/README.md).

### Basics

- Code is located in `src/tau2/domains/`
- Data is located in `data/tau2/domains/`
- Each domain has its own configuration and task definitions

#### View domain-specific policy and API docs:
Run the following command to see the domain policy and API documentation.
```bash
tau2 env <domain>
```

Then visit http://127.0.0.1:8004/redoc

### Environment CLI (beta)

An interactive command-line interface for directly querying and testing domain environments. Features:
- Interactive query interface with domain-specific tools
- Support for multiple domains (airline, mock, etc.)
- Session management with history

To use:
```bash
make env-cli
```

Available commands:
- `:q` - quit the program
- `:d` - change domain
- `:n` - start new session (clears history)

Example usage:
```bash
$ make env-cli

Welcome to the Environment CLI!
Connected to airline domain.

Query (:n new session, :d change domain, :q quit)> What flights are available from SF to LA tomorrow?
Assistant: Let me check the flight availability for you...
[Flight details will appear here]
```

The Environment CLI is useful for:
- Testing domain tools and queries
- Debugging environment responses
- Exploring available domain functionality
- Quick domain interaction without starting the full server stack


## Run tests
To run the test suite use the command

```sh
make test
```

## Config

To configure the framework, see the [config](src/tau2/config.py) file.

### LLM Calls caching
LLM call caching is disabled by default.

To enable LLM calls caching:
    - Make sure `redis` is running.
    - Update the redis config in `config.py` if necessary.
    - Set `LLM_CACHE_ENABLED` to `True` in `config.py`


## Evaluate Your Own Agent
For local or remote agent evaluation, see our [agent developer guide](src/tau2/agent/README.md).

## Contributing

We welcome contributions to τ²-bench! Whether you're fixing bugs, adding new features, creating new domains, or contributing experimental research code, please see our [Contributing Guide](CONTRIBUTING.md) for detailed guidelines on:

- **Opening issues** before starting work
- **Branch naming conventions** and development workflow  
- **Code quality standards** and testing requirements
- **Pull request guidelines** for clean, reviewable contributions
- **Domain and experimental contributions** specific guidelines

For experimental features and research code, check out the [`@experiments/`](src/experiments/) directory.

## Orchestration Sequence Diagram

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant A as Agent
    participant U as UserSimulator
    participant E as Environment

    Note over O: Initialize(task)
    rect rgb(100, 150, 150)
        O->>A: get_init_state_info(message_history)
        A->>O: agent_state_info
        O->>U: get_init_state_info(message_history)
        U->>O: user_state_info
        O->>E: set_state(initialization_data, initialization_actions, message_history)
    end
    Note over O: Start simulation
    loop Pass messages between Agent, User, and Environment

        alt Agent/Env to User
            rect rgb(200, 150, 150)
            O->>U: generate_next_message(msg, user_state_info)
            U-->>O: (user_msg, user_state_info)
            end
            Note over O: Check if user_msg is STOP
        else User/Env to Agent
            rect rgb(100, 200, 100)
            O->>A: generate_next_message(msg, agent_state_info)
            A-->>O: (assistant_msg, agent_state_info)
            Note over O: Check if too many errors
            end
        else User/Agent to Environment
            rect rgb(150, 150, 200)
            O->>E: get_response(tool_call)
            E-->>O: tool_message
            end
        end
        Note over O: Check if max turns reached.
    end
    Note over O: Return simulation run
```

## Citation

```bibtex
@misc{barres2025tau2,
      title={$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment}, 
      author={Victor Barres and Honghua Dong and Soham Ray and Xujie Si and Karthik Narasimhan},
      year={2025},
      eprint={2506.07982},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.07982}, 
}
```
