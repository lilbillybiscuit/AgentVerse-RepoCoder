import asyncio
from enum import Enum
from typing import Any, Dict, List, Tuple, Union
import re
from colorama import Fore

from agentverse.environments import BaseEnvironment
from agentverse.agents.base import BaseAgent
from agentverse.logging_ import logger
from agentverse.message import Message, SolverMessage, ExecutorMessage


from .. import env_registry as EnvironmentRegistry

from agentverse.environments.coderagent.rules import CoderagentRule
from agentverse.utils import extract_id_number


@EnvironmentRegistry.register("coderagent-basic")
class BasicEnvironment(BaseEnvironment):
    rule: CoderagentRule
    agents: Dict[str, Union[BaseAgent, List[BaseAgent]]] = None

    task_description: str

    cnt_turn: int = 0
    max_turn: int = 10
    success: bool = False

    def __init__(self, **kwargs):
        rule_config = kwargs.pop("rule", {})
        planning_config = rule_config.pop(
            "planning", {"type": "multimasking"}
        )
        code_completion_config = rule_config.pop("code_completion", {"type": "basic"})
        rule = CoderagentRule(
            planning_config=planning_config,
            code_completion_config=code_completion_config,
        )
        super().__init__(rule=rule, **kwargs)


    async def step(
        self, group_input, completion_result
    ):
        result = ""
        logs = []

        logger.info(f"Loop Round {self.cnt_turn}")

        # ================== Planning ==================
        # planning = <ID #>
        #planning = await self.rule.planning(
        #    self.task_description, self.agents, self.cnt_turn, group_input, completion_result
        #)
        #logs.append({"module": "Order Planning", "content": planning})
        #logger.info("", f"Next to be completed:\n{planning}", Fore.CYAN)
        planning = None
        pattern = r'<MASK (\d+)>'
        id = None
        try:
            id = int(extract_id_number(planning))
            if (id>=len(completion_result)) or (not re.search(pattern, completion_result[id])):
                id = None
            print(id)
        except:
            logs.append({"module": "Order Planning Error", "content": 'error when extract_id_number'})
            id = None
        print(id, len(completion_result))
        if id==None:
            id = 0
            for completion in completion_result:
                if re.search(pattern, completion):
                    break
                id += 1
        print(id, len(completion_result))
        # ================== Planning ==================

        # ================== Completion ==================
        code_completion, completion_logs = await self.rule.code_completion(
            self.task_description, self.agents, group_input, completion_result, id
        )
        logs.append({"module": "Code Competion", "content": code_completion})
        logs.append({"module": f"Loop Round {self.cnt_turn}", "code completion log": completion_logs})
        logger.info("", f"Completed Code:\n{code_completion}", Fore.YELLOW)
        # ================== Completion ==================
        print(id)
        completion_result[id] = code_completion
        self.success = True
        pattern = r'<MASK (\d+)>'
        for completion in completion_result:
            if re.search(pattern, completion):
                self.success = False
        self.cnt_turn += 1
        return completion_result, logs

    def iter_agents(self):
        for role, agent_or_agents in self.agents.items():
            if isinstance(agent_or_agents, list):
                for agent in agent_or_agents:
                    yield role, agent
            else:
                yield role, agent_or_agents

    def get_spend(self):
        total_spent = sum([agent.get_spend() for (_, agent) in self.iter_agents()])
        return total_spent

    def is_done(self):
        """Check if the environment is done"""
        return self.success or self.cnt_turn>3

    def set_task_description(self, task_description: str = ""):
        self.task_description = task_description

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        self.rule.reset()
