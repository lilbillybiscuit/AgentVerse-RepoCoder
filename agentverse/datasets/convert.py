import json

import json

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as JSON and append to the list
            data.append(json.loads(line))
    return data

def dump_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)
# Example usage
file_path = '/gpfs/gibbs/pi/gerstein/xt86/coderagent/AgentVerse/agentverse/datasets/api_level_completion_1k_context_codegen.test.jsonl'
jsonl_data = load_jsonl(file_path)
res = []
for item in jsonl_data:
    item['metadata']['fpath_info'] = item['metadata']['fpath_tuple']
    prompt_group = [{'gt':item['metadata']['ground_truth'], 'prefix':item['prompt'],'surfix':'','metadata':item['metadata']}]
    res.append({"prompt_group":prompt_group})

file_path = file_path.replace('jsonl','json')
dump_json(res,file_path)