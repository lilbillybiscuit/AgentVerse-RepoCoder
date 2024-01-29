import json
import ast
import openai
from string import Template
from colorama import Fore
from aiohttp import ClientSession
from copy import deepcopy
from typing import TYPE_CHECKING, Any, List, Tuple

from agentverse.agents import NavigatorAgent, DriverAgent
from agentverse.message import Message, NavigatorMessage, DriverMessage
from agentverse.logging_ import logger

from . import BaseCodeCompletion, code_completion_registry
import asyncio
from agentverse.llms.utils.jsonrepair import JsonRepair
import re

@code_completion_registry.register("basic")
class BasicCodeCompletion(BaseCodeCompletion):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    async def astep(
        self,
        navigator: NavigatorAgent,
        driver: DriverAgent,
        task_description: str,
        group_input,
        completion_result,
        id,
        *args,
        **kwargs,
    ):
        stage_description = 'Complete the masked code.'
        item = group_input['prompt_group'][id]
        target = completion_result[id]
        pattern = r'<MASK (\d+)>'
        completion_status = re.search(pattern, target) == None
        assert not completion_status
        metadata = item['metadata']
        prefix = item['prefix']
        surfix = item['surfix']
        file_path = '/'.join(metadata['fpath_tuple'])
        code_input = f"<ID {id}> \n #You can find this code snippet in {file_path}. \n '''python\n{prefix}\n{target}\n{surfix}'''"
        completion = await driver.astep(stage_description=stage_description, code_input=code_input)
        return completion.content, ""

    def broadcast_messages(self, agents, messages) -> None:
        for agent in agents:
            agent.add_message_to_memory(messages)
