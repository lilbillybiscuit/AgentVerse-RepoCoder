from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union, Optional

from agentverse.agents.base import BaseAgent
from agentverse.utils import AGENT_TYPES

from agentverse.environments import BaseRule

from agentverse.environments.coderagent.rules.planning import (
    BasePlanning,
    planning_registry,
)
from agentverse.environments.coderagent.rules.code_completion import (
    BaseCodeCompletion,
    code_completion_registry,
)

class CoderagentRule(BaseRule):
    planning_maker : BasePlanning
    code_completion_maker : BaseCodeCompletion
    def __init__(
        self,
        planning_config,
        code_completion_config,
        *args,
        **kwargs,
    ):
        def build_components(config: Dict, registry):
            component_type = config.pop("type")
            component = registry.build(component_type, **config)
            return component

        planning_maker = build_components(planning_config, planning_registry)
        code_completion_maker = build_components(code_completion_config, code_completion_registry)
        super().__init__(
            planning_maker = planning_maker,
            code_completion_maker = code_completion_maker,
            *args,
            **kwargs,
        )

    async def planning(
        self,
        task_description: str,
        agents: List[BaseAgent],
        cnt_turn: int,
        group_input,
        completion_result
    ) -> str:
        planning = await self.planning_maker.astep(
                agent=agents['navigator'],
                task_description=task_description,
                group_input = group_input,
                completion_result = completion_result
            )
        return planning

    async def code_completion(
        self,
        task_description: str,
        agents: List[BaseAgent],
        group_input,
        completion_result,
        id : int
    ) -> str:
        completion = await self.code_completion_maker.astep(
                navigator=agents['navigator'],
                driver=agents['driver'],
                task_description=task_description,
                group_input = group_input,
                completion_result = completion_result,
                id = id
        )
        return completion


    def reset(self) -> None:
        self.planning_maker.reset()
        self.code_completion_maker.reset()
