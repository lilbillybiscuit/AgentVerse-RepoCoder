# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import editdistance
from collections import defaultdict

from utils import Tools
import json

def compute_EM(target, predictions, passk):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    EM_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip("```").strip("python").strip() for line in prediction.splitlines() if (line.strip("```").strip("python").strip())][:len(target_lines)]
        if len(target_lines) != len(prediction_lines):
            EM_scores.append(0)
            continue
        if target_lines == prediction_lines:
            EM_scores.append(1)
            continue
        EM_scores.append(0)
    return any(EM_scores)

def compute_ES(target, predictions, passk):
    target_lines = [line.strip()  for line in target.splitlines() if line.strip()]
    target_str = '\n'.join(target_lines)
    ES_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip("```").strip("python").strip() for line in prediction.splitlines() if (line.strip("```").strip("python").strip())][:len(target_lines)]
        prediction_str = '\n'.join(prediction_lines)
        ES_scores.append(
            1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))
        )
    return max(ES_scores)

def compute_score_by_repo_with_metadata(repos, lines, stype, passk=1):
    scores = defaultdict(list)
    for line in lines:
        repo = line['metadata']['task_id'].split('/')[0]
        if repo not in repos:
            continue
        #samples = [line['choices'][i]['text'] for i in range(len(line['choices']))]
        samples = [line['choices'][i] for i in range(len(line['choices']))]
        if stype == 'EM':
            score = compute_EM(line['metadata']['ground_truth'], samples, passk)
        elif stype == 'ES':
            score = compute_ES(line['metadata']['ground_truth'], samples, passk)
        scores[repo].append(score)
    avg_scores = {repo: round(sum(scores[repo]) / len(scores[repo]), 4) for repo in scores}
    repo_count = {repo: len(scores[repo]) for repo in scores}
    print(stype)
    score = 0
    cnt = 0
    for repo in avg_scores.keys():
        print(f'{avg_scores[repo]}\t{repo_count[repo]}\t{repo}')
        score += avg_scores[repo] * repo_count[repo]
        cnt += repo_count[repo]
    print(score/cnt)

if __name__ == '__main__':
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
    '''compute single prediction'''
    file_path = '/gpfs/gibbs/pi/gerstein/xt86/coderagent/AgentVerse/agentverse/predict_results/None/longGen_sim_line3_ws30_leaveout1_seperate_rand_basic_1.json'
    data = json.load(open(file_path))# Tools.load_jsonl(file_path)
    print(len(data))
    compute_score_by_repo_with_metadata(repos, data, 'EM', passk=1)
    compute_score_by_repo_with_metadata(repos, data, 'ES', passk=1)
    
