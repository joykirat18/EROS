# # %%
# import pandas as pd
# test_data = pd.read_csv('/home/rohan19095/BTP/PPO/summarizationDataset/test.csv')

# # %%
# test_data.keys()

# # %%
# # import os
# # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# # os.environ["CUDA_VISIBLE_DEVICES"]="1"

# # %%


# # %%
# text = test_data['Original_Text'][0]

# # %%
# from transformers import AutoTokenizer
# path = '/home/rohan19095/BTP/PPO/bartFinetuned2/checkpoint-500'
# bartTokenizer = AutoTokenizer.from_pretrained(path)

# from transformers import AutoModelForSeq2SeqLM

# # s
# model = AutoModelForSeq2SeqLM.from_pretrained(path).to('cuda:0')




# # %%
# import torch
# stateDict = torch.load('/home/rohan19095/BTP/SpanNer-Final/ppo_R2R3/stateDict_2.pth')

# # %%
# stateDict.keys()

# # %%
# model.load_state_dict(stateDict)

# # %%
# from joblib import Parallel, delayed
# from functools import reduce
# from operator import add
# from tqdm import tqdm
# import torch


# class TransformersBaseTokenizer:

#     """Class for encoding and decoding given texts for transformers"""

#     def __init__(
#         self,
#         pretrained_tokenizer,
#         model_type='bart',
#         **kwargs
#         ):
#         self._pretrained_tokenizer = pretrained_tokenizer
#         self.max_seq_len = pretrained_tokenizer.model_max_length
#         self.model_type = model_type
#         self.pad_token_id = pretrained_tokenizer.pad_token_id

#     def __call__(self, *args, **kwargs):
#         return self

#     def tokenizer(self, t):
#         """Limits the maximum sequence length and add the special tokens"""

#         if self.model_type == 'bart':
            
#             CLS = self._pretrained_tokenizer.cls_token
#             SEP = self._pretrained_tokenizer.sep_token
            
#             tokens = \
#                 self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len
#                 - 2]
#             tokens = [CLS] + tokens + [SEP]

#         elif self.model_type == 'pegasus':
#             eos = self._pretrained_tokenizer.eos_token
#             tokens = \
#                 self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len
#                 - 2]
#             tokens = tokens + [eos]
                    

#         return tokens

#     def _numercalise(self, t):
#         """Convert text to their corresponding ids"""
        
#         tokenized_text = self._pretrained_tokenizer(
#                 t,
#                 max_length=self.max_seq_len,
#                 return_tensors='pt',
#                 padding='max_length',
#                 truncation=True,
#                 add_special_tokens=True,
#                 is_split_into_words=False,
#                 )
#         return tokenized_text

#     def _textify(self, input_ids):
#         """Convert ids to their corresponding text"""

#         text = self._pretrained_tokenizer.batch_decode(input_ids,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=False)
#         return text

#     def _chunks(self, lst, n):
#         """Splitting the text into batches"""

#         for i in range(0, len(lst), n):
#             yield lst[i:i + n]

#     def numercalise(self, t, batch_size=4):
#         """Convert text to their corresponding ids and get the attention mask to differentiate between pad and input texts"""

#         if isinstance(t, str):
#             t = [t]  # convert str to list of str

#         n_cores = min(batch_size, os.cpu_count())
#         results = Parallel(n_jobs=n_cores)(delayed(self._numercalise)(batch)
#                 for batch in tqdm(list(self._chunks(t,
#                 batch_size))))
#         input_ids = []
#         attention_masks = []
#         for i in results:
#             input_ids.append(i['input_ids'])
#             attention_masks.append(i['attention_mask'])

#         return {'input_ids': torch.cat(input_ids),
#                 'attention_mask': torch.cat(attention_masks)}

#     def textify(self, tensors, batch_size):
#         """Convert ids to their corresponding text"""

#         if len(tensors.shape) == 1:
#             tensors = [tensors]  # convert 1d tensor to 2d

#         n_cores = min(batch_size, os.cpu_count())
#         results = Parallel(n_jobs=-1, backend='threading'
#                            )(delayed(self._textify)(summary_ids)
#                              for summary_ids in
#                              tqdm(list(self._chunks(tensors,
#                              batch_size))))

#         return reduce(add, results)

# # %%
# tokenizer = TransformersBaseTokenizer(bartTokenizer)

# # %%
# # inputs

# # %%
# # attention_mask

# # %%
# generated_summary = []
# gold_summary = test_data['Summary']
# from tqdm import tqdm
# for text in tqdm(test_data['Original_Text']):
#     inputs = tokenizer._numercalise(text).input_ids.to('cuda:0')
#     attention_mask = tokenizer._numercalise(text).attention_mask.to('cuda:0')
#     outputs = model.generate(inputs,attention_mask=attention_mask,min_new_tokens=200, max_new_tokens=1024, top_p=0.9, do_sample=False)
#     # print(bartTokenizer.decode(outputs[0], skip_special_tokens=True))
#     generated_summary.append(bartTokenizer.decode(outputs[0], skip_special_tokens=True))
    

# # %%
# gold_summary = gold_summary.tolist()

# # %%
# dict = {'policy' : test_data['Original_Text'], 'gold_summary' : gold_summary, 'generated_summary' : generated_summary}
# import pandas as pd
# df = pd.DataFrame(dict)
# df.to_csv('finetune_ppo_ppoR2R3_Results.csv', index=False)


# # %%
# # import pandas as pd
# # df = pd.read_csv('/home/rohan19095/BTP/ctrl-sum/ctrlsum-cnndm_result.csv')

# # %%
# # gold_summary = df['gold_summary'].tolist()
# # generated_summary = df['generated_summary'].tolist()

# # %%
# # from rouge import Rouge

# # def calculate_rouge_scores(gold_summaries, generated_summaries):
# #     rouge = Rouge()
# #     rouge_scores = rouge.get_scores(generated_summaries, gold_summaries, avg=True)

# #     return rouge_scores

# # # Example usage:
# # gold_summaries = [
# #     "The quick brown fox jumps over the lazy dog.",
# #     "The sky is blue."
# # ]
# # generated_summaries = [
# #     "The brown fox jumps over the dog.",
# #     "The sky looks blue."
# # ]

# # rouge_score = calculate_rouge_scores(gold_summary, generated_summary)
# # print(f"ROUGE: {rouge_score}")


# # # %%
# # from evaluate import load
# # bertscore = load("bertscore")
# # # predictions = ["hello there", "general kenobi"]
# # # references = ["hello there", "general kenobi"]
# # results = bertscore.compute(predictions=generated_summary, references=gold_summary, lang="en")

# # # %%
# # sum = 0
# # for f1 in results['f1']:
# #     sum += f1
# # sum = sum / len(results['f1'])
# # print(sum)

# # # %%
# # df

# # # %%
# # from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# # # Reference sentences
# # references = [['the cat is on the mat'],
# #               ['there is a cat on the mat'],
# #               ['the mat has a cat']]

# # # Candidate sentences
# # candidates = ['the cat is on the mat',
# #               'a cat sits on the mat',
# #               'the mat is under the cat']

# # # Create a SmoothingFunction object
# # smoothie = SmoothingFunction().method4

# # # Calculate corpus BLEU score
# # corpus_bleu_score = corpus_bleu(references, candidates, smoothing_function=smoothie)

# # print("Corpus BLEU Score:", corpus_bleu_score)


# # # %%
# # bleu_1 = 0
# # bleu_2 = 0
# # bleu_3 = 0
# # bleu_4 = 0
# # count = 0
# # # import nltk
# # from nltk.translate.bleu_score import sentence_bleu
# # from nltk.translate.meteor_score import meteor_score
# # met = 0
# # weights_1 = (1./1.,)
# # weights_2 = (1./2. , 1./2.)
# # weights_3 = (1./3., 1./3., 1./3.)
# # weights_4 = (1./4., 1./4., 1./4., 1./4.)
# # for i in range(len(gold_summary)):
# #         reference = gold_summary[i].split()
# #         hypothesis = generated_summary[i].split()
# #         bleu_1 += sentence_bleu([reference], hypothesis, weights_1) 
# #         bleu_2 += sentence_bleu([reference], hypothesis, weights_2)
# #         bleu_3 += sentence_bleu([reference], hypothesis, weights_3)
# #         bleu_4 += sentence_bleu([reference], hypothesis, weights_4)
# #         met += meteor_score([reference], hypothesis)
# #         count += 1

# # # %%
# # bleu_1 /= count
# # bleu_2 /= count
# # bleu_3 /= count
# # bleu_4 /= count
# # met /= count

# # # %%
# # print("BLEU-1:", bleu_1)
# # print("BLEU-2:", bleu_2)
# # print("BLEU-3:", bleu_3)
# # print("BLEU-4:", bleu_4)
# # print("Metero:", met)

# # # %%


# # # %%
# # # ROUGE: {'rouge-1': {'r': 0.5505143799376592, 'p': 0.5471579902745551, 'f': 0.531976561164638}, 'rouge-2': {'r': 0.3659014986680469, 'p': 0.36113471117447893, 'f': 0.34730192346188105}, 'rouge-l': {'r': 0.530855935516924, 'p': 0.527336835095412, 'f': 0.5128376189775129}}


# # # %%
# # len(gold_summary), len(generated_summary)

# # # %%
# # gold_summary

# # # %%
# # generated_summary

# # # %%


# # # %%
# # from transformers import pipeline, set_seed, StoppingCriteriaList, StoppingCriteria
# # import torch


# # # %%

# # from transformers import pipeline, set_seed, StoppingCriteriaList, StoppingCriteria
# # import torch
# # class StoppingCriteriaSub(StoppingCriteria):

# #     def __init__(self, tokenizer=None,stops = [], encounters=1):
# #         super().__init__()
# #         self.stops = stops
# #         self.tokenizer=tokenizer

# #     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
# #         # print(input_ids)
# #         full_sentence = self.tokenizer.decode(input_ids[0].tolist(), clean_up_tokenization_spaces=True)
# #         last_token = self.tokenizer.decode(input_ids[0][-1].tolist(), clean_up_tokenization_spaces=True)
# #         import re
# #         pattern = r'\[(\w+)\]'
# #         sequence = re.findall(pattern, full_sentence)
# #         # print(sequence)
# #         # if(len(sequence) > 0 and sequence[-1] != 'find' and last_token == '#'):
# #         #     return True
# #         # if(last_token == '<endoftext>'):
# #         #     return True
# #         print(last_token)
# #         return False

# # # %%
# # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer=tokenizer,stops=['stop_words'])])


# # # %%


# # # %%
# # tokenizer.decode(outputs[0], skip_special_tokens=True)

# # # %%
# # import pandas as pd
# # df = pd.read_csv('finetuneResults.csv')

# # # %%
# # df

# # # %%
# # min_length = 1000000000
# # index = 0
# # for i in range(len(df['gold_summary'].tolist())):
# #     if(min_length > len(df['gold_summary'])):
# #         min_length = len(df['gold_summary'])
# #     # min_length = min(len(df['gold_summary']), min_length)
# #         index = i

# # # %%
# # min_length

# # # %%
# # index

# # # %%
# # import torch
# # model = torch.load('/home/rohan19095/BTP/SpanNer-Final/bart_model_with_loss_penalty.h5')

# # # %%
# # print(df['policy'].tolist()[0])

# # # %%
# # ####Privacy Policy. Harmony Public Schools##Privacy Policy##What information do we collect?##We collect information from you when you respond to a survey or fill out a form. ##What do we use your information for?##Any of the information we collect from you may be used in one of the following ways: ##• To improve our website##(we continually strive to improve our website offerings based on the information and feedback we receive from you)##Do we use cookies?##We do not use cookies.##Do we disclose any information to outside parties?##We do not sell, trade, or otherwise transfer to outside parties your personally identifiable information. This does not include trusted third parties who assist us in operating our website, conducting our business, or servicing you, so long as those parties agree to keep this information confidential. We may also release your information when we believe release is appropriate to comply with the law, enforce our site policies, or protect ours or others rights, property, or safety. However, non-personally identifiable visitor information may be provided to other parties for marketing, advertising, or other uses.##Third party links##Occasionally, at our discretion, we may include or offer third party products or services on our website. These third party sites have separate and independent privacy policies. We therefore have no responsibility or liability for the content and activities of these linked sites. Nonetheless, we seek to protect the integrity of our site and welcome any feedback about these sites.##Online Privacy Policy Only##This online privacy policy applies only to information collected through our website and not to information collected offline.##Your Consent##By using our site, you consent to our online privacy policy.##Changes to our Privacy Policy##If we decide to change our privacy policy, we will post those changes on this page. ##Contacting Us##If there are any questions regarding this privacy policy you may contact us using the information below. ##http://www.harmonytx.org##9431 W. Sam Houston Pkwy S. Houston, TX 77099##U.S.A.

# # # %%
# # print(df['generated_summary'].tolist()[0])

# # # %%
# # print(df['gold_summary'].tolist()[0])

# # # %%
# # We collect information from you when you respond to a survey or fill out a form. ##What do we collect from you may be used in one of the following ways: ##• To improve our website. ##Contacting Us##If there are any questions regarding this privacy policy you may contact us using the information below. ##Online Privacy Policy Only##This online privacy policy applies only to information collected through our website and not to Information collected offline.

# # # %%
# # We collect information from you when you respond to a survey or fill out a form. ##What do we use your information for?##Any of the information we collect from you may be used in one of the following ways: ##• To improve our website##(we continually strive to improve our Website offerings based on the information and feedback we receive from you)##Do we use cookies?##We do not use cookies. ##Third party links##Occasionally, at our discretion, we may include or offer third party products or services on our website. ##Contacting Us##If there are any questions regarding this privacy policy, you may contact us using the information below. ##If we decide to change our Privacy Policy, we will post those changes on this page. ##


# # # %%
# # import json
# # with open('/home/rohan19095/BTP/SpanNer-Final/data/fepd_filtered/spanner.dev', 'rb') as f:
# #     data = json.load(f)

# # # %%
# # with open('span.json', 'w') as f:
# #     json.dump(data,f)

# # # %%
# # import pandas as pd
# # data = pd.read_csv('/home/rohan19095/BTP/PPO/finetune_ppo_ppo_Results.csv')

# # # %%
# # data

# # # %%
# # gold_summary = data['gold_summary'].tolist()
# # generated_summary = data['generated_summary'].tolist()

# # # %%
# # total_ratio = 0
# # for i in range(len(gold_summary)):
# #     total_ratio += (len(generated_summary[i]) / len(gold_summary[i]))



# # # %%
# # total_ratio / len(gold_summary)

# # # %%
# # 1.1434550990241727
# # 0.7335036229111629
# # 0.7691281343967125

# # # %%
# # import pandas as pd

# # # %%
# # train = pd.read_csv('summarizationDataset/train.csv')
# # test = pd.read_csv('summarizationDataset/test.csv')

# # # %%
# # data = pd.concat([train, test], axis=0)

# # # %%
# # from sklearn.model_selection import train_test_split


# # # %%
# # train, dev = train_test_split(data, test_size=0.2, random_state=42)
# # dev, test = train_test_split(dev, test_size=0.5, random_state=42)

# # # %%
# # train.to_csv('summarizationDataset_split/train.csv', index=False)
# # dev.to_csv('summarizationDataset_split/dev.csv', index=False)
# # test.to_csv('summarizationDataset_split/test.csv', index=False)

# # %%

import pandas as pd
df = pd.read_csv('/home/rohan19095/BTP/PPO/finetune_ppo_ppoR2R3_Results.csv')
gold_summary = df['gold_summary'].tolist()
generated_summary = df['generated_summary'].tolist()

from evaluate import load
bertscore = load("bertscore")
# predictions = ["hello there", "general kenobi"]
# references = ["hello there", "general kenobi"]
results = bertscore.compute(predictions=generated_summary, references=gold_summary, lang="en")

sum = 0
for f1 in results['f1']:
    sum += f1
sum = sum / len(results['f1'])
print(sum)



