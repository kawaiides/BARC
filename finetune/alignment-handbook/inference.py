# from transformers import AutoModelForCausalLM
# from peft import PeftModel
# import torch

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("./data/barc-7b-sft-qlora-v0.0.2")

# base_model = AutoModelForCausalLM.from_pretrained(
#     "mistralai/Mistral-7B-v0.3", torch_dtype=torch.float16, 
#     device_map="cuda:0", low_cpu_mem_usage=True,
#     attn_implementation="flash_attention_2",
# )


# peft_model_id = "./data/barc-7b-sft-qlora-v0.0.2"
# model = PeftModel.from_pretrained(base_model, peft_model_id, adapter_name="sft")
# model = torch.compile(model)

# The model is now ready to be used for inference

# get jsonl file from ../arc_problems_train.jsonl

# BASE_MODEL = 'barc0/heavy-barc-llama3.1-8b-ins-fft-transduction_lr1e-5_epoch3'
import os

current_path = os.getcwd()
print("Current working directory:", current_path)

BASE_MODEL = 'llama-3-1-arc-heavy-induction-8b'

LORA_DIR = None
# LORA_DIR = 'barc0/heavy-barc-llama3.1-8b-instruct-lora64-testtime-finetuning'

BATCH_SIZE = 1
BEST_OF = 2

# How many gpus you are using
TENSOR_PARALLEL = 2

from transformers import AutoTokenizer
if LORA_DIR:
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
else:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

import json
data = []
problem_file = 'arc-agi_test_challenges_formatted.jsonl'

import datetime
datetime_str = datetime.datetime.now().strftime("%m%d%H%M%S%f")

with open(problem_file) as f:
    for line in f:
        data.append(json.loads(line))

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

if LORA_DIR:
    llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=64, max_model_len=12000,
            enable_prefix_caching=True, tensor_parallel_size=TENSOR_PARALLEL, dtype='float16')
    lora_request=LoRARequest("barc_adapter", 1, LORA_DIR)
    saving_file = f"{problem_file.replace('.jsonl', '')}_{LORA_DIR.split('/')[-1]}_{datetime_str}.jsonl"
    print(f"Saving to {saving_file}")
else:
    llm = LLM(model=BASE_MODEL, enable_lora=False, max_model_len=16000,
            enable_prefix_caching=True, tensor_parallel_size=TENSOR_PARALLEL, dtype='float16')
    lora_request = None
    if 'checkpoint' in BASE_MODEL.split('/')[-1]:
        model_name = BASE_MODEL.split('/')[-2] + "_" + BASE_MODEL.split('/')[-1]
    else:
        model_name = BASE_MODEL.split('/')[-1]
    saving_file = f"{problem_file.replace('.jsonl', '')}_{model_name}_{datetime_str}.jsonl"

print('batch size:', BATCH_SIZE)

from transformers import StoppingCriteria

class CodeListStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        ss = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        flag = True        
        for s in ss:
            # if "```" occurs twice in the string, then we have a code block
            if s.count("```") < 2:
                flag = False
        
        return flag

from transformers import StoppingCriteriaList
sc = CodeListStoppingCriteria(tokenizer)
scl = StoppingCriteriaList([sc])

responses = []
from tqdm import tqdm
for d in tqdm(data):
    messages = d["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    inputs = tokenizer.apply_chat_template([
        {"role":"system", "content":messages[0]["content"]},
        {"role":"user", "content":messages[1]["content"]}
    ], tokenize=True, return_tensors="pt", add_generation_prompt=True).to("cuda")

    
    outputs = llm.generate(inputs, max_new_tokens=1024, num_return_sequences=1,
        stopping_criteria=scl, temperature=0.2, do_sample=True)

    
    text = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    
    print(text[0])
    # breakpoint()
    responses.append({"uid": d["uid"], "responses": text})

with open("ft_lora_mistral.jsonl", "w") as f:
    for response in responses:
        f.write(json.dumps(response) + "\n")
