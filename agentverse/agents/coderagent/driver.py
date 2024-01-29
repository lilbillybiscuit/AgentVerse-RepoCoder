from __future__ import annotations

import asyncio
from colorama import Fore

from agentverse.logging_ import get_logger
import bdb
from string import Template
from typing import TYPE_CHECKING, List, Tuple

# from agentverse.environments import PipelineEnvironment
from agentverse.message import DriverMessage, Message, NavigatorMessage

from agentverse.agents import agent_registry
from agentverse.agents.base import BaseAgent
from agentverse.utils import AgentCriticism

logger = get_logger()


@agent_registry.register("driver")
class DriverAgent(BaseAgent):
    max_history: int = 5

    def step(
        self, stage_description: str, code_input: str, former_completion: str ='', feedback: str = '', **kwargs
    ) -> DriverMessage:
        pass

    async def astep(
        self, stage_description: str, code_input: str, former_completion: str ='No completion yet', feedback: str = 'No feedback yet', **kwargs
    ) -> DriverMessage:
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
        parsed_response = None
        for i in range(self.max_retry):
            try:
                response = await self.llm.agenerate_response(
                    prepend_prompt, history, append_prompt
                )
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

        message = DriverMessage(
            content=""
            if parsed_response is None
            else parsed_response.return_values["output"],
            sender=self.name,
            receiver=self.get_receiver(),
        )
        return message

    def _fill_prompt_template(
        self,
        former_solution: str,
        critic_opinions: List[AgentCriticism],
        advice: str,
        task_description: str,
    ) -> str:
        """Fill the placeholders in the prompt template

        In the role_assigner agent, three placeholders are supported:
        - ${task_description}
        - ${former_solution}
        - ${critic_messages}
        - ${advice}
        """
        input_arguments = {
            "task_description": task_description,
            "former_solution": former_solution,
            "critic_opinions": "\n".join(
                [
                    f"{critic.sender_agent.role_description} said: {critic.criticism}"
                    for critic in critic_opinions
                ]
            ),
            "advice": advice,
        }
        # if discussion_mode:
        #     template = Template(self.prompt_template[1])
        # else:
        template = Template(self.prompt_template)
        return template.safe_substitute(input_arguments)

    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)

    def reset(self) -> None:
        """Reset the agent"""
        self.memory.reset()
        # TODO: reset receiver
