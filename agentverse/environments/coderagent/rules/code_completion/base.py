from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Tuple, Any

from pydantic import BaseModel

from agentverse.agents import NavigatorAgent
from agentverse.message import SolverMessage, NavigatorMessage

from . import code_completion_registry


class BaseCodeCompletion(BaseModel):
    """
    The base class of planning.
    """

    def step(
        self,
        agent: NavigatorAgent,
        task_description: str,
        solution: List[SolverMessage],
        *args,
        **kwargs,
    ) -> List[NavigatorMessage]:
        pass

    async def astep(
        self,
        agent: NavigatorAgent,
        task_description: str,
        solution: List[str],
        *args,
        **kwargs,
    ) -> List[NavigatorMessage]:
        pass

    def reset(self):
        pass


@code_completion_registry.register("none")
class NoneCodeCompletion(BaseCodeCompletion):
    """
    The base class of execution.
    """

    def step(
        self,
        agent: NavigatorAgent,
        task_description: str,
        solution: List[SolverMessage],
        *args,
        **kwargs,
    ) -> Any:
        return [NavigatorMessage(content="")]
    
    async def astep(
        self,
        agent: NavigatorAgent,
        task_description: str,
        solution: List[SolverMessage],
        *args,
        **kwargs,
    ) -> Any:
        return [NavigatorMessage(content="")]

    def reset(self):
        pass


@code_completion_registry.register("dummy")
class DummyCodeCompletion(BaseCodeCompletion):
    """
    The base class of execution.
    """

    def step(
        self,
        agent: NavigatorAgent,
        task_description: str,
        solution: List[SolverMessage],
        *args,
        **kwargs,
    ) -> Any:
        return [NavigatorMessage(content=s.content) for s in solution]
    
    async def astep(
        self,
        agent: NavigatorAgent,
        task_description: str,
        solution: List[SolverMessage],
        *args,
        **kwargs,
    ) -> Any:
        return [NavigatorMessage(content=s.content) for s in solution]

    def reset(self):
        pass
