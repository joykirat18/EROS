from trl import PPOTrainer,PPOConfig,AutoModelForSeq2SeqLMWithValueHead, create_reference_model, set_seed
from tqdm import tqdm
from torch import nn
# import pickle
# from lion_pytorch import Lion
# import argparse
# import time
import os
# from collections import namedtuple
# from typing import Dict
from collections import namedtuple
# import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
# from torch import Tensor
# from torch.utils.data import DataLoader
# from transformers import AdamW
# from torch.optim import SGD
from rouge import Rouge

# from dataloaders.dataload import BERTNERDataset
# from dataloaders.truncate_dataset import TruncateDataset
# from dataloaders.collate_functions import collate_to_max_length
# from models.bert_model_spanner import BertNER
# from models.config_spanner import BertNerConfig
# from utils.get_parser import get_parser
from radom_seed import set_random_seed
from eval_metric import getEntitiesPredicted
import random
import logging
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)
set_random_seed(0)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from trainer import BertNerTagger, configure_optimizers, get_Custom_dataloader
from joblib import Parallel, delayed
from functools import reduce
from operator import add
from tqdm import tqdm
import torch


class TransformersBaseTokenizer:

    """Class for encoding and decoding given texts for transformers"""

    def __init__(
        self,
        pretrained_tokenizer,
        model_type='bart',
        **kwargs
        ):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.model_max_length
        self.model_type = model_type
        self.pad_token_id = pretrained_tokenizer.pad_token_id

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t):
        """Limits the maximum sequence length and add the special tokens"""

        if self.model_type == 'bart':
            
            CLS = self._pretrained_tokenizer.cls_token
            SEP = self._pretrained_tokenizer.sep_token
            
            tokens = \
                self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len
                - 2]
            tokens = [CLS] + tokens + [SEP]

        elif self.model_type == 'pegasus':
            eos = self._pretrained_tokenizer.eos_token
            tokens = \
                self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len
                - 2]
            tokens = tokens + [eos]
                    

        return tokens

    def _numercalise(self, t):
        """Convert text to their corresponding ids"""
        
        tokenized_text = self._pretrained_tokenizer(
                t,
                max_length=self.max_seq_len,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                add_special_tokens=True,
                is_split_into_words=False,
                )
        return tokenized_text

    def _textify(self, input_ids):
        """Convert ids to their corresponding text"""

        text = self._pretrained_tokenizer.batch_decode(input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
        return text

    def _chunks(self, lst, n):
        """Splitting the text into batches"""

        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def numercalise(self, t, batch_size=4):
        """Convert text to their corresponding ids and get the attention mask to differentiate between pad and input texts"""

        if isinstance(t, str):
            t = [t]  # convert str to list of str

        n_cores = min(batch_size, os.cpu_count())
        results = Parallel(n_jobs=n_cores)(delayed(self._numercalise)(batch)
                for batch in tqdm(list(self._chunks(t,
                batch_size))))
        input_ids = []
        attention_masks = []
        for i in results:
            input_ids.append(i['input_ids'])
            attention_masks.append(i['attention_mask'])

        return {'input_ids': torch.cat(input_ids),
                'attention_mask': torch.cat(attention_masks)}

    def textify(self, tensors, batch_size):
        """Convert ids to their corresponding text"""

        if len(tensors.shape) == 1:
            tensors = [tensors]  # convert 1d tensor to 2d

        n_cores = min(batch_size, os.cpu_count())
        results = Parallel(n_jobs=-1, backend='threading'
                           )(delayed(self._textify)(summary_ids)
                             for summary_ids in
                             tqdm(list(self._chunks(tensors,
                             batch_size))))

        return reduce(add, results)
def loadNERModel():
    parser = BertNerTagger.get_parser()

    parser = Trainer.add_argparse_args(parser)
    # parser.add_argument("--neg_span_weight", type=float, default=0.5,
                            # help="range: [0,1.0], the weight of negative span for the loss.")

    parser.add_argument("--sentence", type=str, default='')
    parser.add_argument('--seed', type=int, default=88888)
    parser.add_argument("--model_name", default="gpt2", type=str)
    parser.add_argument("--toknizer_name", default="gpt2", type=str)
    parser.add_argument("--stateDict_path", default="gpt2", type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    # parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--mini_batch_size", default=2, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--warmup", default=0.1, type=float)
    parser.add_argument("--learning_rate", default=5.41e-6, type=float)
    parser.add_argument("--init_kl_coef", default=0.1, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--input_text_path", default='pseudoCode-Dataset/pseudoCode_csv_full/', type=str)

    parser.add_argument("--save", type=str)


    args = parser.parse_args()

    label2idx = {'O': 0, 'target_direct': 1, 'source_direct': 2, 'source_indirect': 3, 'data': 4,'reason': 5, 'data_compulsory': 6, 'medium': 7, 'target_in_direct': 8, 'data_optional': 9}

    label2idx_list = []
    for lab, idx in label2idx.items():
        pair = (lab, idx)
        label2idx_list.append(pair)
    args.label2idx_list = label2idx_list
    # end{add label2indx augument into the args.}

    # begin{add case2idx augument into the args.}
    morph2idx_list = []
    morph2idx = {'isupper': 1, 'islower': 2,
                 'istitle': 3, 'isdigit': 4, 'other': 5}
    for morph, idx in morph2idx.items():
        pair = (morph, idx)
        morph2idx_list.append(pair)
    args.morph2idx_list = morph2idx_list
    # end{add case2idx augument into the args.}

    args.default_root_dir = args.default_root_dir+'_'+args.random_int

    if not os.path.exists(args.default_root_dir):
        os.makedirs(args.default_root_dir)

    fp_epoch_result = args.default_root_dir+'/epoch_results.txt'
    args.fp_epoch_result = fp_epoch_result

    text = '\n'.join([hp for hp in str(args).replace(
        'Namespace(', '').replace(')', '').split(', ')])
    # print(text)

    text = '\n'.join([hp for hp in str(args).replace(
        'Namespace(', '').replace(')', '').split(', ')])
    fn_path = args.default_root_dir + '/' + args.param_name+'.txt'
    if fn_path is not None:
        with open(fn_path, mode='w') as text_file:
            text_file.write(text)

    model = BertNerTagger(args)
    model = model.to(args.device)
    print(model)
    state_dict = torch.load('/home/rohan19095/BTP/SpanNer-Final/train_logs/fepd/spanner_bert-large-uncased_spMLen_usePruneFalse_useSpLenTrue_useSpMorphFalse_SpWtFalse_value0.5_39820447/spanner_bert-large-uncased_spMLen_usePruneFalse_useSpLenTrue_useSpMorphFalse_SpWtFalse_value0.5best_model_dev.pth',map_location='cuda:0')
    # print(state_dict.keys())
    model.load_state_dict(state_dict,strict=False)

    return model, args

def getPredictedNER(model, args,sentence):
    
    test_dataloader = get_Custom_dataloader(args, sentence)

    # OPTIMIZER
    # optimizers, schedulers = configure_optimizers(model, args)
    model.eval()
    outputs = []
    nerEntities = []
    try:
        with torch.no_grad():
            for batch in tqdm(test_dataloader):

                output = {}

                tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = batch
                loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken,
                        all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs]

                attention_mask = (tokens != 0).long()
                all_span_rep, all_span_feature_rep = model.forward(
                    loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids)
                predicts = model.classifier(all_span_rep)

                predicts = predicts.to(args.device)
                span_label_ltoken = span_label_ltoken.to(args.device)
                real_span_mask_ltoken = real_span_mask_ltoken.to(args.device)

                entities = getEntitiesPredicted(args, all_span_word, words, predicts, span_label_ltoken,
                                            all_span_idxs)
                # print(words)
                # print(entities)
                nerEntities.extend(entities)

                output['entities_preds'] = entities

                outputs.append(output)
        # print(sentence)
        # print(nerEntities)
        return nerEntities
    except Exception as e:
        print(e)
        print("ERROR in NER")
        # print(sentence)
        # print(nerEntities)
        return nerEntities

import pandas as pd
import torch
def combinetext(path):

    combine = []
    df = pd.read_csv(path)
    df = df.sample(frac = 1)
    originalText = df['Original_Text'].tolist()
    summary = df['Summary'].tolist()
    # for index, row in df.iterrows():
    #     prompt = row['Original_Text']
    #     pseudoCode = row['Summary']
    #     text = prompt + ' <sep> ' + pseudoCode + ' <|endoftext|>'
    #     combine.append(text)

    return originalText, summary

from torch.utils.data import Dataset
class TextRLDataset(Dataset):
    def __init__(self, input_texts, labelTexts, tokenizer):
        self.input_text = input_texts
        self.label_text = labelTexts
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.input_text)
    
    def __getitem__(self, idx):
        input_text = self.input_text[idx]
        label_text = self.label_text[idx]
        # text = self.texts[idx]
        # encoded_text = self.tokenizer.encode_plus(
        #     text,
        #     max_length=self.tokenizer.model_max_length,
        #     add_special_tokens=True,
        #     padding='max_length',
        #     truncation=True,
        #     return_tensors='pt'
        # )
        # input_ids = encoded_text['input_ids'].squeeze()
        # attention_mask = encoded_text['attention_mask'].squeeze()
        return {
            'input_text': input_text,
            'label_text': label_text,
        }
    
def getDataset(path, tokenizer):
    originalText, summary = combinetext(path)
    dataset=TextRLDataset(originalText,summary, tokenizer)
    return dataset
    
def build_data_pipeline(args, tokenizer):
    input_text_path = args.input_text_path
    # from utils.pseudoCodeDataloader import getDataset
    train_data = getDataset(input_text_path + 'train.csv', tokenizer)
    valid_data = getDataset(input_text_path + 'test.csv', tokenizer)

    return train_data, valid_data

def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def load_augmented_model (args):

    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.model_name,low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.toknizer_name)
    import torch
    print(model)
    stateDict = torch.load('/home/rohan19095/BTP/SpanNer-Final/bart_with_loss.pth')
    print(model.pretrained_model.load_state_dict(stateDict))
    # print(model)
    model_ref = create_reference_model(model)

    model.to('cuda')
    model.train()

    return model,model_ref,tokenizer

from transformers import pipeline, set_seed, StoppingCriteriaList, StoppingCriteria
import torch
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, tokenizer=None,stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.tokenizer=tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # print(input_ids)
        full_sentence = self.tokenizer.decode(input_ids[0].tolist(), clean_up_tokenization_spaces=True).strip()
        last_token = self.tokenizer.decode(input_ids[0][-1].tolist(), clean_up_tokenization_spaces=True).strip()
        import re

        if(last_token == '<|endoftext|>'):
            return True
        return False
    
def generate(model, tokenizer, prompt,target,k=10,p=0.7,output_length=120,temperature=1,num_return_sequences=1,repetition_penalty=1.3):
    # model.to('cuda')
    # model.eval()
    tokenizer_2 = TransformersBaseTokenizer(tokenizer)
    encoded_prompt = tokenizer_2._numercalise(prompt).input_ids[0]
    attention_mask = tokenizer_2._numercalise(prompt).attention_mask[0]
    # encoded_prompt = tokenizer.encode(prompt,max_length=tokenizer.model_max_length, truncation=True, add_special_tokens=True,return_tensors='pt')[0]
    query_tensor = encoded_prompt
    # if(encoded_prompt.shape[0] >= tokenizer.model_max_length):
        # response_text = "NO RESPONSE GENERATED"
        # response_tensor = tokenizer.encode(response_text, add_special_tokens=True, return_tensors="pt")[0]
        # breakpoint()
        # return query_tensor, response_tensor, response_text
    
    # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer=tokenizer,stops=stop_words)])
    # breakpoint()
    with torch.no_grad():
        output_sequence = model.generate(
            # input_ids=encoded_prompt,
            encoded_prompt.to('cuda'),
            # attention_mask=attention_mask.to('cuda'),
            min_new_tokens=203,
            max_new_tokens=1024,
            # num_beams=5,
            # early_stopping="never",
            # top_k=10,
            top_p=0.9,
            # temperature=1,
            # num_beams=5,
            # repetition_penalty=0.2,
            do_sample=False,
            # stopping_criteria=stopping_criteria
        
        )

    output_sequence = output_sequence[0].tolist()
    text = tokenizer.decode(output_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    # print(text)
    # text = text[:text.find('<|endoftext|>')].strip()
    # if('<sep>' in text):
    #     text = text[text.find('<sep>') + 5:].strip()
    # else:
    #     text = text[len(prompt + ' <sep>'):].strip()
    response_tensor = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")[0]
    # text = tokenizer.decode(response_tensor, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    print("generated")
    print(text)
    print()
    print("response")
    print(target)

    return query_tensor, response_tensor, text


def saveModelBest(args, ppo_trainer):
    print("Saving best model")
    os.makedirs(args.save + '/best', exist_ok=True)

    ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).save_pretrained(args.save + '/best')
    # torch.save(ppo_trainer.model.state_dict(), args.save +'/best' +  '/stateDict.pth')
    torch.save(ppo_trainer.model.pretrained_model.state_dict(), args.save + '/best' + '/stateDict_2.pth')

def saveModel(args, ppo_trainer):
    print("Saving model")
    os.makedirs(args.save, exist_ok=True)

    ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).save_pretrained(args.save)
    # torch.save(ppo_trainer.model.state_dict(), args.save + '/stateDict.pth')
    torch.save(ppo_trainer.model.pretrained_model.state_dict(), args.save + '/stateDict_2.pth')

def getReward(model, args,generated_summary, originalPolicy, goldSummary):
    print("RUNNING REWARD")
    from collections import Counter

    gold_entity = getPredictedNER(model, args, goldSummary)
    generated_entity = getPredictedNER(model, args, generated_summary)
    # print("entities predicted")
    generated_counter = Counter(generated_entity)
    gold_counter = Counter(gold_entity)
    # print(generated_counter)
    # print(gold_counter)
    total_reward = 0

    for entity, count in gold_counter.items():
        generated_count = generated_counter.get(entity, 0)
        reward = min(generated_count, count)
        total_reward += reward


    max_reward = sum(gold_counter.values())
    # print(total_reward)
    # print(max_reward)
    try:
        # import math
        R1 = (total_reward - 0.2 * abs(sum(generated_counter.values()) - total_reward)) / max_reward
        # R1 = total_reward / max_reward
    except Exception as e:
        print(e)
        print("Error in reward 1")
        R1 = 0

    
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, goldSummary)
    rouge_1 = scores[0]['rouge-1']['f']
    rouge_2 = scores[0]['rouge-2']['f']
    rouge_l = scores[0]['rouge-l']['f']
    # R2 = (rouge_1 + rouge_2 + rouge_l) / 3
    R2 = rouge_l

    R3 = 1 - (abs(len(generated_summary) - len(goldSummary))/ max(len(goldSummary),len(generated_summary)))
    print(R1, R2, R3)
    reward = (R1 + R2 + R3)
    # reward = (R2)min



    return torch.tensor(float(reward))



def main():

    model, args = loadNERModel()
    import pandas as pd
    # df = pd.read_csv('/home/rohan19095/BTP/PPO/finetuneResults.csv')
    # gold_summary = df['gold_summary'].tolist()
    # generated_summary = df['generated_summary'].tolist()
    # from tqdm import tqdm
    # gold_ner = []
    # generated_ner = []

    # for summary in tqdm(gold_summary, desc="first"):
    #     gold_ner.append(getPredictedNER(model, args, summary))
    # for summary in tqdm(generated_summary, desc="second"):
    #     generated_ner.append(getPredictedNER(model, args, summary))

    # df = pd.read_csv('/home/rohan19095/BTP/PPO/finetune_loss_Results.csv')
    # generated_summary = df['generated_summary'].tolist()
    # generated_loss_ner = []
    # for summary in tqdm(generated_summary,desc="third"):
    #     generated_loss_ner.append(getPredictedNER(model, args, summary))

    # df = pd.read_csv('/home/rohan19095/BTP/PPO/finetune_ppo_ppo9_Results.csv')
    # generated_summary = df['generated_summary'].tolist()
    # generated_ppo_ner = []
    
    # for summary in tqdm(generated_summary, desc="last"):
    #     generated_ppo_ner.append(getPredictedNER(model, args, summary))

    # train_df = pd.read_csv('/home/rohan19095/BTP/PPO/summarizationDataset_split/train.csv')
    # document = train_df['Original_Text'].tolist()
    # train_summary = train_df['Summary'].tolist()
    # from tqdm import tqdm
    # document_ner = []
    train_ner = []
    document_dev_ner = []
    document_test_ner = []
    # for summary in tqdm(document, desc="document"):
    #     document_ner.append(getPredictedNER(model, args, summary))

    # dict1 = {'document': document_ner, 'train' : train_ner, 'dev' : dev_ner, 'test' : test_ner}

    # import json
    # with open('nerResult_summary.json', 'w') as f:
    #     json.dump(dict1, f)

    # for summary in tqdm(train_summary, desc="train_ner"):
    #     train_ner.append(getPredictedNER(model, args, summary))

    # dict1 = {'document': document_ner, 'train' : train_ner, 'dev' : dev_ner, 'test' : test_ner}

    # import json
    # with open('nerResult_summary.json', 'w') as f:
    #     json.dump(dict1, f)
    train_df = pd.read_csv('/home/rohan19095/BTP/PPO/summarizationDataset_split_full2/train.csv')
    train_document = train_df['Original_Text'].tolist()
    
    for document in tqdm(train_document, desc="train_summary"):
        train_ner.append(getPredictedNER(model, args, document))
    
    dev_df = pd.read_csv('/home/rohan19095/BTP/PPO/summarizationDataset_split_full2/dev.csv')
    dev_document = dev_df['Original_Text'].tolist()
    
    for document in tqdm(dev_document, desc="dev_summary"):
        document_dev_ner.append(getPredictedNER(model, args, document))
    

    test_df = pd.read_csv('/home/rohan19095/BTP/PPO/summarizationDataset_split_full2/test.csv')
    test_document = test_df['Original_Text'].tolist()
    
    for document in tqdm(test_document, desc="test_summary"):
        document_test_ner.append(getPredictedNER(model, args, document))

    dict1 = {'train': train_ner, 'dev' : document_dev_ner, 'test' : document_test_ner}

    import json
    with open('nerResult_document.json', 'w') as f:
        json.dump(dict1, f)
    


    
    # dict1 = {'gold_ner':gold_ner, 'generated_ner': generated_ner, 'generated_loss_ner' : generated_loss_ner, 'generated_ppo_ner' : generated_ppo_ner}
    
    # with open('/home/rohan19095/BTP/PPO/finetuneResults.csv', 'r') as f:
        # data = json.load(f)
    # newEntities = []
    
    # model, model_ref, tokenizer = load_augmented_model(args)
    # print(model)
    # train_data, test_data = build_data_pipeline(args, tokenizer)

    # config = PPOConfig(
    #     model_name="summarization-bart",
    #     learning_rate=args.learning_rate,
    #     init_kl_coef=args.init_kl_coef,
    #     gamma=args.gamma,
    #     # adap_kl_ctrl=False,
    #     # kl_penalty='mse',
    #     batch_size=8,
    #     mini_batch_size=4,
    #     optimize_cuda_cache=True,

    #     # log_with='wandb',
    #     # ratio_threshold=1000,
    #     seed=113431)
    # print("Loading ppo model...")
    # ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer, dataset=train_data)

    # del model
    # del model_ref
    # torch.cuda.empty_cache() 

    # num_epochs = 4
    # from tqdm import tqdm
    # best_acc = 0

    # print("Training PPO model")
    # baseReward = 0
    # max_reward = 0
    # for epoch in tqdm(range(num_epochs)):
    #     print("Epoch: ", epoch)
    #     counter = 0
    #     for _, data_batch in tqdm(enumerate(ppo_trainer.dataloader)):
    #         # try:
    #         input_text = data_batch['input_text']
    #         label_text = data_batch['label_text']
    #         # input_ids = data_batch['input_ids']
    #         # attention_masks = data_batch['attention_mask']
    #         # print(input_ids)
    #         # print(sentence)
    #         response_tensors = []
    #         response_texts = []
    #         question_texts = []
    #         gold_answers = []
    #         query_tensors = []
    #         query_texts = []
    #         gold_response_texts = []
    #         rewards = []
    #         pointer = 0
    #         for i in tqdm(range(len(input_text))):

    #             # sentence = query
    #             policy = input_text[i]
    #             target = label_text[i]
    #             # prompt = sentence[: sentence.find('<sep>')].strip()
    #             # target = sentence[sentence.find('<sep>')+5:]
    #             # target = target[: target.find('<endoftext>')]

    #             XtoYInput = policy
    #             query_tensor, response_tensor, response_text = generate(ppo_trainer, tokenizer, XtoYInput, target)
    #             print("len of response tensor: ", len(response_tensor))
    #             query_tensors.append(query_tensor)
    #             response_tensors.append(response_tensor)
    #             reward = getReward(nerModel, args, response_text, policy, target)
    #             print("reward: ",reward)
    #             rewards.append(reward)

    #             print()

    #             response_texts.append(response_text)
    #             query_texts.append(target)
    #             pointer += 1


    #         print("rewards: ", rewards)
    #         reward_sum = 0
    #         for r in rewards:
    #             reward_sum += r.item()
    #         if(max_reward < reward_sum):
    #             saveModelBest(args, ppo_trainer)
    #             max_reward = reward_sum
    #         if(counter == 0):
    #             baseReward = reward_sum
    #         print("reward sum: ", reward_sum)
    #         print("Base reward: ", baseReward)
    #         if(reward_sum > baseReward):
    #             saveModel(args, ppo_trainer)
    #         stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    #         # try:
    #         #     ppo_trainer.log_stats(stats, {'query' : [],'response' :[]}, rewards)
    #         # except Exception as e:
    #         #     print("ERROR IN WANDB log")
    #         #     print(e)
    #         #     continue
    #         counter += 1
    
    #         # except Exception as e:
    #         #     print(e)
    #         #     # print(sentence)
    #         #     print("Error in training")
    #         #     print("\n")
    #         #     continue
    #     # from utils.generate import getAccuracy
    #     # acc = getAccuracy(ppo_trainer.model, ppo_trainer.tokenizer, '/home/joykirat/JS/GPTJPPO/dataset/svamp.json')
    #     # print("accuracySVAMP: ", acc)
    #     # if(best_acc < acc):
    #         # best_acc = acc
    #         # saveModel()
    


if __name__ == '__main__':
    main()