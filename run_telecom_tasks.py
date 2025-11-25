import argparse
import json
import time
from pathlib import Path
from typing import Optional

from tau2.environment.gym_env import Tau2GymEnv
from tau2.utils.llm_utils import generate


def save_trajectory_to_file(task_id, trial, messages, trajectory, reward, reward_info, trajectory_dir, agent_context=None, env=None):
    """
    将轨迹保存到 JSON 文件。
    
    参数:
        task_id: 任务 ID
        trial: 试验编号
        messages: 来自 orchestrator.trajectory 的消息列表
        trajectory: 步骤观察列表
        reward: 总奖励
        reward_info: 奖励分解信息
        trajectory_dir: 保存轨迹的目录
        agent_context: 代理执行时的上下文（用于复现）
        env: 环境实例（用于获取 user state）
    """
    from datetime import datetime
    
    traj_dir = Path(trajectory_dir)
    traj_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Agent 视角：直接使用执行时记录的上下文
    agent_trace = None
    if agent_context:
        agent_messages = []
        for i, msg in enumerate(agent_context["messages"]):
            # 跳过第一条消息（"Hi! How can I help you today?"）
            # if hasattr(msg, 'content') and msg.content == "Hi! How can I help you today?":
            #     continue
                
            if hasattr(msg, 'model_dump'):
                agent_messages.append(msg.model_dump())
            else:
                agent_messages.append({"role": getattr(msg, 'role', 'unknown'), "content": str(msg)})
        
        # 序列化工具
        agent_tools = []
        for tool in agent_context["tools"]:
            if hasattr(tool, 'model_dump'):
                agent_tools.append(tool.model_dump(mode='json'))
            else:
                agent_tools.append(str(tool))
                
        agent_trace = {
            "system_prompt": agent_context["system_prompt"],
            "tools": agent_tools,
            "messages": agent_messages
        }
    
    # 2. User 视角：从 User Simulator 的内部状态获取
    user_trace = None
    if env and env.orchestrator and env.orchestrator.user_state:
        user_state = env.orchestrator.user_state
        
        # 获取 user messages，并反转 role（因为从 User Simulator 视角，它是 assistant）
        user_messages = []
        for msg in user_state.messages:
            if hasattr(msg, 'model_dump'):
                msg_dict = msg.model_dump()
                # 反转 role：assistant ↔ user
                if msg_dict.get('role') == 'assistant':
                    msg_dict['role'] = 'user'
                elif msg_dict.get('role') == 'user':
                    msg_dict['role'] = 'assistant'
                user_messages.append(msg_dict)
            else:
                role = getattr(msg, 'role', 'unknown')
                # 反转 role
                if role == 'assistant':
                    role = 'user'
                elif role == 'user':
                    role = 'assistant'
                user_messages.append({"role": role, "content": str(msg)})
        
        # 获取 system prompt
        user_system_prompt = ""
        if user_state.system_messages:
            user_system_prompt = user_state.system_messages[0].content
            
        # 获取 tools
        user_tools = env.environment.get_user_tools()
        
        # 序列化工具
        serialized_user_tools = []
        for tool in (user_tools or []):
            if hasattr(tool, 'model_dump'):
                serialized_user_tools.append(tool.model_dump(mode='json'))
            else:
                serialized_user_tools.append(str(tool))
        
        user_trace = {
            "system_prompt": user_system_prompt,
            "tools": serialized_user_tools,
            "messages": user_messages
        }
    
    # 3. 完整对话：第三方观察视角（原始的 orchestrator.trajectory）
    full_conversation = []
    for msg in messages:
        if hasattr(msg, 'model_dump'):
            full_conversation.append(msg.model_dump())
        else:
            full_conversation.append({
                "role": getattr(msg, 'role', 'unknown'),
                "content": str(msg)
            })
    
    # 构建完整轨迹数据
    trajectory_data = {
        "task_id": task_id,
        # "trial": trial,
        "timestamp": datetime.now().isoformat(),
        
        # Agent 复现所需数据（来自执行时记录）
        "messages": agent_trace,
        
        # User 复现所需数据（来自 User Simulator 状态）
        "user_messages": user_trace,
        
        # 完整对话记录
        "full_conversation": full_conversation,
        
        # 步骤级轨迹（observation, action, reward）
        "trajectory": trajectory,
        
        # 评估结果
        "evaluation": {
            "final_reward": reward,
            "reward_info": reward_info.model_dump() if reward_info and hasattr(reward_info, 'model_dump') else reward_info
        }
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_id}_trial{trial}_{timestamp}.json"
    filepath = traj_dir / filename
    
    # 保存到文件
    with open(filepath, 'w') as f:
        json.dump(trajectory_data, f, indent=2, default=str)
    
    return filepath


def create_llm_policy(llm_model: str):
    """
    创建基于 LLM 的策略函数。
    
    参数:
        llm_model: 用于代理的 LLM 模型
    
    返回:
        策略函数
    """
    def policy(env):
        """基于 LLM 的策略。"""
        # 从环境获取工具和策略
        tools = env.environment.get_tools()
        policy_text = env.environment.get_policy()
        
        # 构建系统提示（使用 tau2 的原始格式）
        from tau2.data_model.message import SystemMessage, AssistantMessage, UserMessage, ToolMessage, MultiToolMessage
        from tau2.agent.base import is_valid_agent_history_message
        
        agent_instruction = """You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""
        
        system_prompt = f"""<instructions>
{agent_instruction}
</instructions>
<policy>
{policy_text}
</policy>"""
        
        messages = [SystemMessage(role="system", content=system_prompt)]
        
        # 过滤并展开代理的轨迹消息
        # 代理看到: AssistantMessage, UserMessage (非工具), ToolMessage (发给助手)
        # MultiToolMessage 展开为单独的 ToolMessage
        for msg in env.orchestrator.trajectory:
            if isinstance(msg, MultiToolMessage):
                # 展开 MultiToolMessage - 仅包含发给助手的工具消息
                for tool_msg in msg.tool_messages:
                    if tool_msg.requestor == "assistant":
                        messages.append(tool_msg)
            elif is_valid_agent_history_message(msg):
                messages.append(msg)
        
        # 使用 LLM 生成响应
        action = generate(
            model=llm_model,
            tools=tools,
            messages=messages,
        )
        
        # 返回动作和上下文信息（用于忠实记录）
        context = {
            "messages": messages,
            "tools": tools,
            "system_prompt": system_prompt
        }
        
        return action, context
    
    return policy


def run_single_episode(env, policy_fn, task_id, trial, trajectory_dir=None):
    """
    运行单个回合。
    
    参数:
        env: Tau2GymEnv 实例
        policy_fn: 策略函数
        task_id: 要运行的任务 ID
        trial: 试验编号
        trajectory_dir: 保存轨迹文件的目录
    
    返回:
        包含回合结果的字典
    """
    observation, info = env.reset(options={"task_id": task_id})
    
    print(f"\n{'='*80}")
    print(f"Task: {task_id} | Trial: {trial + 1}")
    print(f"{'='*80}")
    
    total_reward = 0
    step_count = 0
    max_steps = env.max_steps
    
    # 轨迹跟踪
    trajectory = []
    last_agent_context = None
    
    start_time = time.time()
    
    while step_count < max_steps:
        # 从策略获取动作
        # 注意：不传递 observation，因为 policy 从 env.orchestrator.trajectory 获取完整历史
        try:
            action, context = policy_fn(env)
            last_agent_context = context
        except Exception as e:
            print(f"Error generating action: {e}")
            break
        
        # 在执行动作前记录步骤
        step_entry = {
            "step": step_count + 1,
            "action": action.model_dump() if hasattr(action, 'model_dump') else str(action),
            "observation": None,  # 将在步骤后填充
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {}
        }
        
        # 执行步骤
        observation, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # 更新轨迹条目
        step_entry["observation"] = observation.model_dump() if hasattr(observation, 'model_dump') else str(observation)
        step_entry["reward"] = reward
        step_entry["terminated"] = terminated
        step_entry["truncated"] = truncated
        step_entry["info"] = step_info
        trajectory.append(step_entry)
        
        if step_count % 5 == 0:
            print(f"  Step {step_count}/{max_steps}...", end='\r')
        
        if terminated or truncated:
            duration = time.time() - start_time
            print(f"\n  Finished at step {step_info['step_count']}")
            print(f"  Reason: {step_info.get('termination_reason', 'unknown')}")
            print(f"  Reward: {total_reward:.3f}")
            print(f"  Duration: {duration:.1f}s")
            
            if 'reward_info' in step_info:
                reward_info = step_info['reward_info']
                if reward_info.reward_breakdown:
                    print(f"  Breakdown: {reward_info.reward_breakdown}")
            
            # 保存轨迹
            if trajectory_dir:
                save_trajectory_to_file(
                    task_id=task_id,
                    trial=trial,
                    messages=env.orchestrator.trajectory,
                    trajectory=trajectory,
                    reward=total_reward,
                    reward_info=step_info.get('reward_info'),
                    trajectory_dir=trajectory_dir,
                    agent_context=last_agent_context,
                    env=env
                )
            
            return {
                "task_id": task_id,
                "trial": trial,
                "reward": total_reward,
                "steps": step_info['step_count'],
                "termination_reason": step_info.get('termination_reason'),
                "duration": duration,
                "reward_info": step_info.get('reward_info'),
            }
    
    # 达到最大步数但未终止（不应该发生）
    duration = time.time() - start_time
    
    # 保存轨迹
    if trajectory_dir:
        save_trajectory_to_file(
            task_id=task_id,
            trial=trial,
            messages=env.orchestrator.trajectory,
            trajectory=trajectory,
            reward=total_reward,
            reward_info=None,
            trajectory_dir=trajectory_dir,
            agent_context=last_agent_context,
            env=env
        )
    
    return {
        "task_id": task_id,
        "trial": trial,
        "reward": total_reward,
        "steps": step_count,
        "termination_reason": "max_steps",
        "duration": duration,
    }


def run_tasks(
    domain: str = "telecom",
    task_ids: Optional[list[str]] = None,
    num_trials: int = 1,
    agent_llm: str = "gpt-4.1",
    user_llm: str = "gpt-4.1",
    max_steps: int = 100,
    max_errors: int = 10,
    num_tasks: Optional[int] = None,
):
    """
    运行多个任务和多次试验。
    
    参数:
        domain: 域名
        task_ids: 要运行的任务 ID 列表（None = 所有任务）
        num_trials: 每个任务的试验次数
        agent_llm: 代理的 LLM 模型
        user_llm: 用户模拟器的 LLM 模型
        max_steps: 每个回合的最大步数
        max_errors: 每个回合的最大错误数
        num_tasks: 要运行的任务数量（如果未指定 task_ids）
    
    返回:
        包含所有结果的字典
    """
    # 固定路径配置
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trajectory_dir = f"results/telecom_trajectories"
    save_to = f"results/telecom_run_{timestamp}.json"
    
    print("="*80)
    print(f"Tau2 Gymnasium Interface - {domain.upper()} Domain Runner")
    print("="*80)
    print(f"Agent LLM: {agent_llm}")
    print(f"User LLM: {user_llm}")
    print(f"Trials per task: {num_trials}")
    print(f"Max steps: {max_steps}")
    print(f"Saving trajectories to: {trajectory_dir}")
    print(f"Saving results to: {save_to}")
    print("="*80)
    
    # 创建环境
    env = Tau2GymEnv(
        domain=domain,
        task_ids=task_ids,
        llm_user=user_llm,
        max_steps=max_steps,
        max_errors=max_errors,
    )
    
    # 创建策略
    policy = create_llm_policy(agent_llm)
    
    tasks_to_run = [t.id for t in env.tasks]
    if num_tasks is not None:
        tasks_to_run = tasks_to_run[:num_tasks]
    
    print(f"\nRunning {len(tasks_to_run)} tasks × {num_trials} trials = {len(tasks_to_run) * num_trials} episodes\n")
    
    # 运行所有回合
    all_results = []
    total_episodes = len(tasks_to_run) * num_trials
    episode_count = 0
    
    for task_id in tasks_to_run:
        for trial in range(num_trials):
            episode_count += 1
            print(f"\n{'#'*80}")
            print(f"Episode {episode_count}/{total_episodes}")
            print(f"{'#'*80}")
            
            result = run_single_episode(
                env=env,
                policy_fn=policy,
                task_id=task_id,
                trial=trial,
                trajectory_dir=trajectory_dir,
            )
            all_results.append(result)
    
    env.close()
    
    # 计算汇总统计
    rewards = [r['reward'] for r in all_results]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    success_rate = sum(1 for r in rewards if r >= 1.0) / len(rewards) if rewards else 0
    
    summary = {
        "domain": domain,
        "agent_llm": agent_llm,
        "user_llm": user_llm,
        "num_trials": num_trials,
        "total_episodes": len(all_results),
        "avg_reward": avg_reward,
        "success_rate": success_rate,
        "results": all_results,
    }
    
    # 打印摘要
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total episodes: {len(all_results)}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Success rate: {success_rate:.1%} ({sum(1 for r in rewards if r >= 1.0)}/{len(rewards)})")
    print(f"Reward distribution:")
    print(f"  Perfect (1.0): {sum(1 for r in rewards if r >= 1.0)}")
    print(f"  Partial (0.0-1.0): {sum(1 for r in rewards if 0 < r < 1.0)}")
    print(f"  Failed (0.0): {sum(1 for r in rewards if r == 0.0)}")
    print("="*80)
    
    # 保存结果
    save_path = Path(save_to)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 将 reward_info 转换为字典以进行 JSON 序列化
    for result in summary['results']:
        if 'reward_info' in result and result['reward_info']:
            result['reward_info'] = result['reward_info'].model_dump()
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {save_path}")
    
    return summary


def main():
    """主入口点。"""
    parser = argparse.ArgumentParser(
        description="Run telecom tasks with Gymnasium interface",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument(
        "--domain",
        type=str,
        default="telecom",
        help="Domain name (default: telecom)"
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Specific task IDs to run (default: all tasks)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to run (default: all tasks)"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials per task (default: 1)"
    )
    parser.add_argument(
        "--agent-llm",
        type=str,
        default="gpt-4.1",
        help="LLM model for agent (default: gpt-4.1)"
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default="gpt-4.1",
        help="LLM model for user simulator (default: gpt-4.1)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per episode (default: 100)"
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Maximum errors per episode (default: 10)"
    )
    
    args = parser.parse_args()
    
    run_tasks(
        domain=args.domain,
        task_ids=args.task_ids,
        num_trials=args.num_trials,
        agent_llm=args.agent_llm,
        user_llm=args.user_llm,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        num_tasks=args.num_tasks,
    )


if __name__ == "__main__":
    main()
