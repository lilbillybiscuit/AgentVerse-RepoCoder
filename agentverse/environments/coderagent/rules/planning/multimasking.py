import json
import ast
import openai
from string import Template
from colorama import Fore
from aiohttp import ClientSession
from copy import deepcopy
from typing import TYPE_CHECKING, Any, List, Tuple

from agentverse.agents import NavigatorAgent
from agentverse.message import Message, NavigatorMessage, DriverMessage
from agentverse.logging_ import logger

from . import BasePlanning, planning_registry
import asyncio
from agentverse.llms.utils.jsonrepair import JsonRepair
import re


@planning_registry.register("multimasking")
class MultiMaskingPlanning(BasePlanning):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    async def astep(
        self,
        agent: NavigatorAgent,
        task_description: str,
        group_input,
        completion_result,
        *args,
        **kwargs,
    ):
        stage_description = 'Decide next mask to be filled.'
        code_input = []
        id = 0
        for item, target in zip(group_input['prompt_group'], completion_result):
            metadata = item['metadata']
            prefix = item['prefix']
            file_path = '/'.join(metadata['fpath_tuple'])
            pattern = r'<MASK (\d+)>'
            completion_status = re.search(pattern, target) == None
            completion_status = 'Completed' if completion_status else 'Uncompleted'
            code_input.append(f"<ID {id}> ({completion_status}) \n #You can find this code snippet in {file_path}. \n '''python\n{prefix}\n{target}\n'''")
            id += 1
        code_input = '\n'.join(code_input)
        agent.do_planning()
        planning = await agent.astep(stage_description=stage_description, code_input=code_input)
        return planning.content

    def broadcast_messages(self, agents, messages) -> None:
        for agent in agents:
            agent.add_message_to_memory(messages)
