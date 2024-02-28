# %%
import pandas as pd
test_data = pd.read_csv('/home/rohan19095/BTP/PPO/summarizationDataset/test.csv')

# %%
test_data.keys()

# %%
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# %%
# !nvidia-smi


# %%
# text = test_data['Original_Text'][0]

# %%
import sys
from transformers import AutoTokenizer
path = str(sys.argv[1])
tokenizer = AutoTokenizer.from_pretrained(path)

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(path).to('cuda:0')
print(path)

# %%
generated_summary = []
gold_summary = test_data['Summary']
from tqdm import tqdm
for text in tqdm(test_data['Original_Text']):
    inputs = tokenizer(text,max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to('cuda:0')
    outputs = model.generate(inputs, max_new_tokens=1024, do_sample=False)
    generated_summary.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    

# %%
gold_summary = gold_summary.tolist()

# %%
from rouge import Rouge

def calculate_rouge_scores(gold_summaries, generated_summaries):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(generated_summaries, gold_summaries, avg=True)

    return rouge_scores

rouge_score = calculate_rouge_scores(gold_summary, generated_summary)
print(f"ROUGE: {rouge_score}")


# %%
# ROUGE: {'rouge-1': {'r': 0.5505143799376592, 'p': 0.5471579902745551, 'f': 0.531976561164638}, 'rouge-2': {'r': 0.3659014986680469, 'p': 0.36113471117447893, 'f': 0.34730192346188105}, 'rouge-l': {'r': 0.530855935516924, 'p': 0.527336835095412, 'f': 0.5128376189775129}}


# # %%
# len(gold_summary), len(generated_summary)

# # %%
# gold_summary

# # %%
# generated_summary

# # %%


# # %%
# from transformers import pipeline, set_seed, StoppingCriteriaList, StoppingCriteria
# import torch


# # %%

# from transformers import pipeline, set_seed, StoppingCriteriaList, StoppingCriteria
# import torch
# class StoppingCriteriaSub(StoppingCriteria):

#     def __init__(self, tokenizer=None,stops = [], encounters=1):
#         super().__init__()
#         self.stops = stops
#         self.tokenizer=tokenizer

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#         # print(input_ids)
#         full_sentence = self.tokenizer.decode(input_ids[0].tolist(), clean_up_tokenization_spaces=True)
#         last_token = self.tokenizer.decode(input_ids[0][-1].tolist(), clean_up_tokenization_spaces=True)
#         import re
#         pattern = r'\[(\w+)\]'
#         sequence = re.findall(pattern, full_sentence)
#         # print(sequence)
#         # if(len(sequence) > 0 and sequence[-1] != 'find' and last_token == '#'):
#         #     return True
#         # if(last_token == '<endoftext>'):
#         #     return True
#         print(last_token)
#         return False

# # %%
# stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer=tokenizer,stops=['stop_words'])])


# # %%


# # %%
# tokenizer.decode(outputs[0], skip_special_tokens=True)

# # %%



