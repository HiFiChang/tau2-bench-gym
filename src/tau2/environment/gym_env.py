import uuid
from typing import Any, Optional

import gymnasium as gym
from loguru import logger

from tau2.data_model.message import AssistantMessage, Message, ToolCall, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.registry import registry
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator
from tau2.utils.llm_utils import get_cost
from tau2.utils.utils import get_now


class Tau2GymEnv(gym.Env):
    """
    - 代理是学习/决策实体
    - 用户和环境动态从代理的角度来看是"环境"的一部分
    
    动作：AssistantMessage（文本或工具调用）
    观察：代理接收到的消息（UserMessage、ToolMessage 或 MultiToolMessage）
    奖励：由 τ2 评估器计算的回合级奖励（回合期间为 0，结束时为最终值）

    - 一次 gym.step(action) = 代理行动 → 等待控制返回到代理
    - 用户-环境交互隐藏在 step() 内部
    - 保留所有原始 τ2 评估逻辑
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        domain: str,
        task_ids: Optional[list[str]] = None,
        task_set_name: Optional[str] = None,
        llm_user: str = "gpt-4.1",
        llm_args_user: Optional[dict] = None,
        max_steps: int = 100,
        max_errors: int = 10,
        render_mode: Optional[str] = None,
    ):
        """
        初始化 tau2 Gym 环境。
        
        参数:
            domain: 域名（"telecom", "retail", "airline"）
            task_ids: 要使用的特定任务 ID，或 None 使用所有任务
            task_set_name: 任务集名称，默认为域名
            llm_user: 用户模拟器的 LLM 模型
            llm_args_user: 用户 LLM 的参数
            max_steps: 每个回合的最大步数
            max_errors: 允许的最大工具错误数
            render_mode: 渲染模式（仅支持 "human"）
        """
        super().__init__()
        
        self.domain = domain
        self.llm_user = llm_user
        self.llm_args_user = llm_args_user if llm_args_user is not None else {}
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.render_mode = render_mode
        
        self.evaluation_type = EvaluationType.ALL
        
        # 加载任务
        if task_set_name is None:
            task_set_name = domain
        self.tasks = get_tasks(task_set_name=task_set_name, task_ids=task_ids)
        if len(self.tasks) == 0:
            raise ValueError(f"No tasks found for domain {domain}")
        
        self.current_task_idx = 0
        self.current_task: Optional[Task] = None
        
        # 当前回合状态
        self.orchestrator: Optional[Orchestrator] = None
        self.environment = None
        self.user = None
        
        # 回合跟踪
        self.episode_count = 0
        
        # Gym 空间（灵活的，不严格执行）
        self.action_space = None  # AssistantMessage 对于 gym.spaces 来说太复杂
        self.observation_space = None  # Message 对于 gym.spaces 来说太复杂
        
        logger.info(f"Initialized Tau2GymEnv with {len(self.tasks)} tasks from domain '{domain}'")
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> tuple[Message, dict[str, Any]]:
        """
        重置环境以开始新回合。
        
        参数:
            seed: 此回合的随机种子
            options: 附加选项（可以包含 'task_id' 以选择特定任务）
        
        返回:
            observation: 代理的初始消息（用户响应问候后的 UserMessage）
            info: 包含回合元数据的字典
        """
        super().reset(seed=seed)
        
        # 选择任务
        if options and "task_id" in options:
            # 通过 ID 查找任务
            task_id = options["task_id"]
            task = next((t for t in self.tasks if t.id == task_id), None)
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            self.current_task = task
        else:
            # 循环遍历任务
            self.current_task = self.tasks[self.current_task_idx]
            self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
        
        # 创建环境
        environment_constructor = registry.get_env_constructor(self.domain)
        self.environment = environment_constructor()
        
        # 创建用户
        user_tools = self.environment.get_user_tools()
        self.user = UserSimulator(
            tools=user_tools,
            instructions=str(self.current_task.user_scenario),
            llm=self.llm_user,
            llm_args=self.llm_args_user,
        )
        
        # 创建占位符代理，实际上不使用
        # orchestrator需要一个代理对象
        from tau2.agent.base import BaseAgent
        
        class DummyAgent(BaseAgent):
            """占位符代理 - 外部策略将提供动作"""
            def generate_next_message(self, message, state):
                raise NotImplementedError("External policy provides actions")
            
            def get_init_state(self, message_history=None):
                return {"messages": message_history if message_history else []}
            
            def stop(self):
                pass
        
        dummy_agent = DummyAgent()
        
        # 创建orchestrator
        self.orchestrator = Orchestrator(
            domain=self.domain,
            agent=dummy_agent,
            user=self.user,
            environment=self.environment,
            task=self.current_task,
            max_steps=self.max_steps,
            max_errors=self.max_errors,
            seed=seed,
            solo_mode=False,
        )
        
        # 初始化编排器
        self.orchestrator.initialize()
        
        # 此时，orchestrator已向用户发送 "Hi! How can I help you today?"
        # 执行步骤直到控制返回到assistant
        # 初始状态：from_role=AGENT, to_role=USER, message=greeting
        
        # 执行直到我们回到代理
        while self.orchestrator.to_role != Role.AGENT and not self.orchestrator.done:
            self.orchestrator.step()
        
        # 现在 orchestrator.message 包含用户的响应
        observation = self.orchestrator.message
        
        info = {
            "task_id": self.current_task.id,
            "domain": self.domain,
            "episode": self.episode_count,
            "step_count": self.orchestrator.step_count,
        }
        
        self.episode_count += 1
        
        return observation, info
    
    def step(
        self, 
        action: AssistantMessage
    ) -> tuple[Message, float, bool, bool, dict[str, Any]]:
        """
        执行一个代理动作。
        
        参数:
            action: 来自代理的 AssistantMessage（文本或工具调用）
        
        返回:
            observation: 代理的下一条消息
            reward: 回合期间为 0.0，完成时为最终奖励
            terminated: 如果回合自然结束（代理/用户停止）则为 True
            truncated: 如果回合达到限制（max_steps/max_errors）则为 True
            info: 步骤元数据
        """
        if self.orchestrator is None:
            raise RuntimeError("Must call reset() before step()")
        
        # 验证动作
        if not isinstance(action, AssistantMessage):
            raise TypeError(f"Action must be AssistantMessage, got {type(action)}")
        
        try:
            action.validate()
        except Exception as e:
            raise ValueError(f"Invalid action: {e}")
        
        # 将动作注入编排器
        self.orchestrator.message = action
        self.orchestrator.from_role = Role.AGENT
        
        # 根据动作类型确定 to_role
        if action.is_tool_call():
            self.orchestrator.to_role = Role.ENV
        else:
            self.orchestrator.to_role = Role.USER
        
        # 将动作添加到轨迹
        self.orchestrator.trajectory.append(action)
        
        # 执行orchestrator步骤直到控制返回到代理
        max_inner_steps = 100  # 防止无限循环
        inner_step_count = 0
        while (
            self.orchestrator.to_role != Role.AGENT 
            and not self.orchestrator.done 
            and inner_step_count < max_inner_steps
        ):
            self.orchestrator.step()
            inner_step_count += 1
            
            # 检查终止条件
            if self.orchestrator.step_count >= self.max_steps:
                self.orchestrator.done = True
                self.orchestrator.termination_reason = TerminationReason.MAX_STEPS
                break
            
            if self.orchestrator.num_errors >= self.max_errors:
                self.orchestrator.done = True
                self.orchestrator.termination_reason = TerminationReason.TOO_MANY_ERRORS
                break
        
        if inner_step_count >= max_inner_steps:
            logger.warning(f"Hit max inner steps ({max_inner_steps}) - possible infinite loop")
            self.orchestrator.done = True
            self.orchestrator.termination_reason = TerminationReason.MAX_STEPS
        
        # 获取观察
        observation = self.orchestrator.message
        
        # 计算奖励和终止
        if self.orchestrator.done:
            # 构建模拟运行
            simulation_run = self._build_simulation_run()
            
            # 使用 τ2 的评估器进行评估
            reward_info = evaluate_simulation(
                simulation=simulation_run,
                task=self.current_task,
                evaluation_type=self.evaluation_type,
                solo_mode=False,
                domain=self.domain,
            )
            
            reward = reward_info.reward
            
            # 确定 terminated 与 truncated
            terminated = self.orchestrator.termination_reason in [
                TerminationReason.AGENT_STOP,
                TerminationReason.USER_STOP,
            ]
            truncated = not terminated
            
            info = {
                "step_count": self.orchestrator.step_count,
                "termination_reason": self.orchestrator.termination_reason.value,
                "reward_info": reward_info,
                "inner_steps": inner_step_count,
            }
        else:
            reward = 0.0
            terminated = False
            truncated = False
            info = {
                "step_count": self.orchestrator.step_count,
                "from_role": self.orchestrator.from_role.value,
                "to_role": self.orchestrator.to_role.value,
                "inner_steps": inner_step_count,
            }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """渲染当前状态（打印轨迹）。"""
        if self.render_mode != "human":
            return
        
        if self.orchestrator is None:
            print("Environment not initialized. Call reset() first.")
            return
        
        print("\n" + "="*80)
        print(f"Task: {self.current_task.id}")
        print(f"Step: {self.orchestrator.step_count}/{self.max_steps}")
        print("="*80)
        
        for msg in self.orchestrator.trajectory[-5:]:  # 显示最后 5 条消息
            print(f"\n{msg}")
        
        print("="*80 + "\n")
    
    def close(self):
        """清理资源。"""
        self.orchestrator = None
        self.environment = None
        self.user = None
    
    def _build_simulation_run(self) -> SimulationRun:
        """从当前编排器状态构建 SimulationRun。"""
        messages = self.orchestrator.get_trajectory()
        res = get_cost(messages)
        if res is None:
            agent_cost, user_cost = None, None
        else:
            agent_cost, user_cost = res
        
        end_time = get_now()
        
        simulation_run = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.current_task.id,
            start_time=end_time,
            end_time=end_time,
            duration=0.0,  # 不跟踪
            termination_reason=self.orchestrator.termination_reason.value,
            reward_info=None,  # 由 evaluate_simulation 填充
            user_cost=user_cost,
            agent_cost=agent_cost,
            messages=messages,
            seed=self.orchestrator.seed,
        )
        
        return simulation_run
    
    # @staticmethod
    # def make_text_action(content: str) -> AssistantMessage:
    #     """
    #     创建文本消息动作的辅助函数。
        
    #     参数:
    #         content: 来自代理的文本内容
        
    #     返回:
    #         带有文本内容的 AssistantMessage
    #     """
    #     return AssistantMessage(
    #         role="assistant",
    #         content=content,
    #         cost=0.0,
    #     )
    
    # @staticmethod
    # def make_tool_action(tool_calls: list[ToolCall]) -> AssistantMessage:
    #     """
    #     创建工具调用动作的辅助函数。
        
    #     参数:
    #         tool_calls: 要执行的工具调用列表
        
    #     返回:
    #         带有工具调用的 AssistantMessage
    #     """
    #     return AssistantMessage(
    #         role="assistant",
    #         tool_calls=tool_calls,
    #         cost=0.0,
    #     )
    
    # @staticmethod
    # def make_single_tool_action(
    #     tool_name: str, 
    #     arguments: dict,
    #     tool_id: Optional[str] = None
    # ) -> AssistantMessage:
    #     """
    #     创建单个工具调用动作的辅助函数。
        
    #     参数:
    #         tool_name: 要调用的工具名称
    #         arguments: 工具参数
    #         tool_id: 可选的工具调用 ID
        
    #     返回:
    #         带有单个工具调用的 AssistantMessage
    #     """
    #     if tool_id is None:
    #         tool_id = f"call_{uuid.uuid4().hex[:8]}"
        
    #     tool_call = ToolCall(
    #         id=tool_id,
    #         name=tool_name,
    #         arguments=arguments,
    #         requestor="assistant",
    #     )
        
    #     return Tau2GymEnv.make_tool_action([tool_call])
