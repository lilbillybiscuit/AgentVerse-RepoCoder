import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
from torch.multiprocessing import Process, set_start_method
import concurrent
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import os
import random
import functools
from agentverse.repocoder.utils import Tools, FilePathBuilder, CodexTokenizer, CodeGenTokenizer, CONSTANTS
from copy import deepcopy
from agentverse.repocoder.build_prompt import PromptBuilder
from agentverse.coderagent import CoderAgent
#from api_utils import api_handler
random.seed(2024)
import os
os.environ["AZURE_OPENAI_API_KEY"] = "d84fea092ad54d66b2843a1b546ab14c"
os.environ["AZURE_OPENAI_API_BASE"] = "https://biocodeeval-openai.openai.azure.com/"
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_file_path', type=str)
    parser.add_argument('--retrieval_file_path', type=str)
    parser.add_argument('--num_processes', type=int, default=2)
    parser.add_argument('--cuda_devices', type=str, default='[1,2,3]')
    parser.add_argument('--inference_mode', type=str, default='test')
    parser.add_argument('--retrieval_mode', type=str, default='gt')
    parser.add_argument('--plan_mode', type=str, default='rand', choices=['rand', 'anchor_first'])
    parser.add_argument('--benchmark', type=str)
    parser.add_argument('--model_name', default="Salesforce/codegen-2B-mono")
    
    args = parser.parse_args()
    args.cuda_devices = eval(args.cuda_devices)
    return args 
    
def read_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        data_list = [json.loads(line.strip()) for line in file]
    return data_list
    
if __name__ == "__main__":
    try:
        set_start_method("spawn")  # Required for CUDA support in multiprocessing on certain systems
    except RuntimeError:
        pass
    args = parse_args()
    mapping = { CONSTANTS.api_benchmark: 'api_completion_benchmark', CONSTANTS.sim_benchmark: 'sim_assign_completion_benchmark', 
CONSTANTS.sim1_benchmark: 'sim_assign1_completion_benchmark', CONSTANTS.rand_benchmark:'rand_assign_completion_benchmark', CONSTANTS.line_benchmark:'random_line_completion_benchmark', CONSTANTS.short_api_benchmark:'short_api_completion_benchmark', CONSTANTS.short_line_benchmark: 'short_random_line_completion_benchmark' , CONSTANTS.ast_benchmark:'ast_assign_completion_benchmark'}
    if args.retrieval_file_path != None:
        setattr(FilePathBuilder, mapping[args.benchmark], args.retrieval_file_path)
    output_base_path = f'./predict_results/{args.benchmark}/'
    log_base_path = f'./logs/{args.benchmark}/'
    task_file_path = args.task_file_path
    repos = [
        'huggingface_diffusers',
        'nerfstudio-project_nerfstudio',
        'awslabs_fortuna',
        'huggingface_evaluate',
        'google_vizier',
        'alibaba_FederatedScope',
        'pytorch_rl',
        'opendilab_ACE',
    ]
    task_name = task_file_path.split('/')[-1].split('.')[0] + f'_seperate_{args.plan_mode}'
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    if not os.path.exists(log_base_path):
        os.makedirs(log_base_path)
    # Number of processes and CUDA devices
    num_processes = args.num_processes
    cuda_devices = args.cuda_devices
    
    prompts = json.load(open(task_file_path))[0:10] #read_jsonl_file(task_file_path)
    logs = []
    res = []

    for prompt in tqdm(prompts):
        coder_agent = CoderAgent.from_task('coderagent', '/gpfs/gibbs/pi/gerstein/xt86/coderagent/AgentVerse/agentverse/tasks')
        completion, logs_ = coder_agent.run(prompt)
        logs.append(logs_)
        for item, completion_ in zip(prompt['prompt_group'], completion):
            metadata = item['metadata']
            metadata['ground_truth'] = item['gt']
            res.append({'metadata':metadata, 'choices':[completion_]})
        with open(os.path.join(output_base_path,f'{task_name}_3.json'), 'w') as file:
            json.dump(res, file)
        with open(os.path.join(log_base_path,f'{task_name}_3.json'), 'w') as file:
            json.dump(logs, file)
    