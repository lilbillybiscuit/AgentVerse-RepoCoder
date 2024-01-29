from __future__ import annotations

import json
from colorama import Fore
from agentverse.logging_ import get_logger
import bdb
from string import Template
from typing import TYPE_CHECKING, List, Union

from agentverse.message import Message

from agentverse.agents import agent_registry
from agentverse.agents.base import BaseAgent
from agentverse.utils import AgentCriticism
from agentverse.message import NavigatorMessage

from pydantic import BaseModel, Field
logger = get_logger()


@agent_registry.register("navigator")
class NavigatorAgent(BaseAgent):
    max_history: int = 3

    planning_prepend_prompt_template: str = Field(default="")
    planning_append_prompt_template: str = Field(default="")
    critic_prepend_prompt_template: str = Field(default="")
    critic_append_prompt_template: str = Field(default="")
    testing_prepend_prompt_template: str = Field(default="")
    testing_append_prompt_template: str = Field(default="")
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    def do_critic(self):
        self.append_prompt_template = self.critic_append_prompt_template
        self.prepend_prompt_template = self.critic_prepend_prompt_template
    def do_planning(self):
        self.append_prompt_template = self.planning_append_prompt_template
        self.prepend_prompt_template = self.planning_prepend_prompt_template
    def do_testing(self):
        self.append_prompt_template = self.testing_append_prompt_template
        self.prepend_prompt_template = self.testing_prepend_prompt_template
    def step(self, stage_description: str, code_input: str, former_completion: str ='No completion yet', feedback: str = 'No feedback yet', **kwargs) -> NavigatorMessage:
        pass

    async def astep(
        self, stage_description: str, code_input: str, former_completion: str ='No completion yet', feedback: str = 'No feedback yet', **kwargs
    ) -> NavigatorMessage:
        """Asynchronous version of step"""
        logger.debug("", self.name, Fore.MAGENTA)
        prepend_prompt, append_prompt, prompt_token = self.get_all_prompts(
            stage_description=stage_description,
            code_input=code_input,
            former_completion = former_completion,
            feedback = feedback,
            **kwargs,
        )

        max_send_token = self.llm.send_token_limit(self.llm.args.model)
        max_send_token -= prompt_token

        history = await self.memory.to_messages(
            self.name,
            start_index=-self.max_history,
            max_send_token=max_send_token,
            model=self.llm.args.model,
        )
        parsed_response: Union[AgentCriticism, None] = None
        for i in range(self.max_retry):
            try:
                print('call llm')
                response = await self.llm.agenerate_response(
                    prepend_prompt, history, append_prompt
                )
                print('call llm finished')
                parsed_response = self.output_parser.parse(response)
                break
            except (KeyboardInterrupt, bdb.BdbQuit):
                raise
            except Exception as e:
                logger.error(e)
                logger.warn("Retrying...")
                continue

        if parsed_response is None:
            logger.error(f"{self.name} failed to generate valid response.")

        message = NavigatorMessage(
            content=parsed_response.return_values['output'] if parsed_response is not None else "",
            sender=self.name,
            sender_agent=self,
        )
        return message

    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)

    def reset(self) -> None:
        """Reset the agent"""
        self.memory.reset()
        # TODO: reset receiver
