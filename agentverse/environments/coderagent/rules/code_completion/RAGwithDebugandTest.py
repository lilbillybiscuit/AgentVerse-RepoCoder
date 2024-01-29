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
import os
from agentverse.repocoder.utils import Tools, FilePathBuilder, CodexTokenizer, CodeGenTokenizer, CONSTANTS
from copy import deepcopy
from agentverse.repocoder.build_prompt import PromptBuilder
from pydantic import BaseModel, Field

@code_completion_registry.register("rag_with_debug")
class RAGwithDebugCodeCompletion(BaseCodeCompletion):
    retrieval_folder_path : str
    retrieval_file_path : str
    retrieval_mode : str
    retrieval_results_dict: dict = Field(default= {})
    completion_max_turn : int
    prompt_builder: Any = Field(default= None)
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        self.retrieval_results_dict = dict()
        for root, dirs, files in os.walk(self.retrieval_folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                query_lines_with_retrieval_results = Tools.load_pickle(file_path)
                for query_line in query_lines_with_retrieval_results:
                    self.retrieval_results_dict[query_line['metadata']['task_id']] = query_line
        self.prompt_builder = PromptBuilder(self.retrieval_results_dict, self.retrieval_file_path, 'No need to log', CodeGenTokenizer)

    def get_RAG(self, group_input, completion_result, id):
        prompt_group = group_input['prompt_group']
        item = prompt_group[id]
        retrieval_result = self.retrieval_results_dict[item['metadata']['task_id']]
        retrieval_result_ungroup_ = []
        for retrieved_context_ in retrieval_result['top_k_context']:
            retrieved_context_metadatas = retrieved_context_[0]['metadata']
            for retrieved_context_metadata in retrieved_context_metadatas:
                retrieved_context = deepcopy(retrieved_context_[0])
                retrieved_context['metadata'] = [retrieved_context_metadata]
                retrieval_result_ungroup_.append((retrieved_context, retrieved_context_[1]))
        retrieval_result_ungroup = []
        for retrieved_context_ in retrieval_result_ungroup_:
            retrieved_context = retrieved_context_[0]
            retrieved_context_metadatas = retrieved_context['metadata']
            for group_item, completion in zip(prompt_group, completion_result):
                group_item_fpath = os.path.join(*group_item['metadata']['fpath_tuple'])
                for retrieved_context_metadata in retrieved_context_metadatas:
                    if os.path.join(*retrieved_context_metadata['fpath_tuple']) == group_item_fpath:
                        retrieved_context['context'] = retrieved_context['context'].replace(group_item['gt'], completion)
        context_set = set()
        for retrieved_context_ in retrieval_result_ungroup_:
            if retrieved_context_[0]['context'] in context_set:
                continue
            context_set.add(retrieved_context_[0]['context'])
            retrieval_result_ungroup.append(retrieved_context_)
        retrieval_result = retrieval_result_ungroup 
        new_prompt, chosen_context = self.prompt_builder._build_prompt(self.retrieval_mode, item['prefix'], retrieval_result)
        return new_prompt
    
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
        logs = []
        stage_description = 'Complete the masked code.'
        print(id)
        item = group_input['prompt_group'][id]
        target = completion_result[id]
        pattern = r'<MASK (\d+)>'
        completion_status = re.search(pattern, target) == None
        assert not completion_status
        metadata = item['metadata']
        prefix = self.get_RAG(group_input=group_input, completion_result=completion_result,id=id)#item['prefix']
        surfix = item['surfix']
        file_path = '/'.join(metadata['fpath_tuple'])
        code_input = f"<ID {id}> \n #You can find this code snippet in {file_path}. \n '''python\n{prefix}\n{target}\n{surfix}'''"
        completion = "No completion yet"
        feedback = "No feedback yet"
        for i in range(self.completion_max_turn):
            completion = await driver.astep(stage_description=stage_description, code_input=code_input, former_completion=completion, feedback = feedback)
            completion = completion.content
            navigator.do_critic()
            if i == self.completion_max_turn-1:
                break
            feedback = await navigator.astep(stage_description=f'{stage_description} You need to critic the completion given by your partner', code_input=code_input, former_completion=completion, feedback = feedback)
            feedback = feedback.content
            logger.info("", f"Critic on {completion}:\n{feedback}", Fore.CYAN)
            logs.append({"Round": i, "completion":completion, "feedback":feedback})
            if feedback == 'correct':
                break
            tests = await navigator.astep(stage_description=f'{stage_description} You need to generate unit tests for the code completion given by your partner', code_input=code_input,function_description=item['metadata'], function_name=item['prefix'], former_completion=competion, test=test)
            tests = tests.content
            navigator.do_testing()
            logger.info("", f"Testing on {completion}:\n{tests}", Fore.CYAN)
        logs.append({"Round": i, "completion":completion, "feedback":feedback})
        return completion, logs

    def broadcast_messages(self, agents, messages) -> None:
        for agent in agents:
            agent.add_message_to_memory(messages)
