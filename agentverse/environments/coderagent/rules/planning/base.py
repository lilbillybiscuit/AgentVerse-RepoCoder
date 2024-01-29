from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Tuple, Any

from pydantic import BaseModel

from agentverse.agents import NavigatorAgent
from agentverse.message import SolverMessage, NavigatorMessage

from . import planning_registry


class BasePlanning(BaseModel):
    """
    The base class of planning.
    """

    def step(
        self,
        agent: NavigatorAgent,
        task_description: str,
        group_input,
        completion_result,
        *args,
        **kwargs,
    ):
        pass

    async def astep(
        self,
        agent: NavigatorAgent,
        task_description: str,
        group_input,
        completion_result,
        *args,
        **kwargs,
    ):
        pass

    def reset(self):
        pass


@planning_registry.register("none")
class NonePlanning(BasePlanning):
    """
    The base class of execution.
    """

    def step(
        self,
        agent: NavigatorAgent,
        task_description: str,
        group_input,
        completion_result,
        *args,
        **kwargs,
    ):
        return [NavigatorMessage(content="")]
    
    async def astep(
        self,
        agent: NavigatorAgent,
        task_description: str,
        group_input,
        completion_result,
        *args,
        **kwargs,
    ):
        return [NavigatorMessage(content="")]

    def reset(self):
        pass


@planning_registry.register("dummy")
class DummyPlanning(BasePlanning):
    """
    The base class of execution.
    """

    def step(
        self,
        agent: NavigatorAgent,
        task_description: str,
        group_input,
        completion_result,
        *args,
        **kwargs,
    ):
        return [NavigatorMessage(content=s.content) for s in solution]
    
    async def astep(
        self,
        agent: NavigatorAgent,
        task_description: str,
        group_input,
        completion_result,
        *args,
        **kwargs,
    ):
        return [NavigatorMessage(content=s.content) for s in solution]

    def reset(self):
        pass
