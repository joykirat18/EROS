# %%
from trl import PPOTrainer,PPOConfig, AutoModelForCausalLMWithValueHead,AutoModelForSeq2SeqLMWithValueHead, create_reference_model, set_seed
# from ppo_trainer_diffValue import PPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
import torch
import pandas as pd 
import os
from peft import LoraConfig, TaskType, get_peft_model
from utils.pseudoCodeDataloader import getDataset
import argparse
from utils.reward import getReward


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=88888)
parser.add_argument("--model_name", default="gpt2", type=str)
parser.add_argument("--toknizer_name", default="gpt2", type=str)
parser.add_argument("--stateDict_path", default="gpt2", type=str)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--mini_batch_size", default=1, type=int)
parser.add_argument("--num_train_epochs", default=10, type=int)
parser.add_argument("--warmup", default=0.1, type=float)
parser.add_argument("--learning_rate", default=1.41e-6, type=float)
parser.add_argument("--init_kl_coef", default=0.03, type=float)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--input_text_path", default='pseudoCode-Dataset/pseudoCode_csv_full/', type=str)

parser.add_argument("--save", type=str)

args, _ = parser.parse_known_args()

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
    
# %%
def load_augmented_model ():


    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name,low_cpu_mem_usage=True)
    stateDict = torch.load('/home/rohan19095/BTP/SpanNer-Final/bart_with_loss.pth')
    model.pretrained_model.load_state_dict(stateDict)
    # model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(args.toknizer_name)
    # model.pretrained_model.resize_token_embeddings(len(tokenizer))
    # model.pretrained_model.load_state_dict(torch.load(args.stateDict_path))
    print(model)
    model_ref = create_reference_model(model)

    model.to('cuda')
    model.train()

    return model,model_ref,tokenizer

import pandas as pd
import torch
def combinetext(path):

    combine = []
    df = pd.read_csv(path)
    for index, row in df.iterrows():
        prompt = row['Original_Text']
        pseudoCode = row['Summary']
        text = prompt + ' <sep> ' + pseudoCode + ' <|endoftext|>'
        combine.append(text)

    return combine


from torch.utils.data import Dataset
class TextRLDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=self.tokenizer.model_max_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_text['input_ids'].squeeze()
        attention_mask = encoded_text['attention_mask'].squeeze()
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

def getDataset(path, tokenizer):
    text = combinetext(path)
    dataset=TextRLDataset(text, tokenizer)
    return dataset



def build_data_pipeline(tokenizer):
    input_text_path = args.input_text_path
    # from utils.pseudoCodeDataloader import getDataset
    train_data = getDataset(input_text_path + 'train.csv', tokenizer)
    valid_data = getDataset(input_text_path + 'test.csv', tokenizer)

    return train_data, valid_data

# %%
print("Loading model...")
model, model_ref,tokenizer = load_augmented_model()

# %%
def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# %%
import random
seed = random.randint(1, 1000)
print("SEED: ", seed)
set_seed(seed)

# %%
print("Building data pipeline...")
train_data, test_data = build_data_pipeline(tokenizer)


# %%
config = PPOConfig(
    model_name="summarization-bart",
    learning_rate=args.learning_rate,
    init_kl_coef=args.init_kl_coef,
    gamma=args.gamma,
    batch_size=args.batch_size,
    mini_batch_size=args.mini_batch_size,
    optimize_cuda_cache=True,
    seed=args.seed)


# %%
print("Loading ppo model...")
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer, dataset=train_data)
# lr_scheduler=lr_scheduler
del model
del model_ref
torch.cuda.empty_cache()    

# %%
from transformers import pipeline, set_seed, StoppingCriteriaList, StoppingCriteria
import torch


# %%

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
        pattern = r'\[(\w+)\]'
        sequence = re.findall(pattern, full_sentence)
        # print(sequence)
        if(len(sequence) > 0 and sequence[-1] != 'find' and last_token == '#'):
            return True
        if(last_token == '<endoftext>'):
            return True
        return False
    
def generate(model, tokenizer, prompt,target,k=0,p=0.99,output_length=120,temperature=1,num_return_sequences=1,repetition_penalty=1.0):
    model.to('cuda')
    model.eval()
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True,return_tensors='pt')
    stop_words = ["<endoftext>"]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer=tokenizer,stops=stop_words)])

    with torch.no_grad():
        output_sequence = model.generate(
            # input_ids=encoded_prompt,
            input_ids=encoded_prompt.to('cuda'),
            max_new_tokens=output_length,
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            stopping_criteria=stopping_criteria
        
        )

    output_sequence = output_sequence[0].tolist()
    text = tokenizer.decode(output_sequence, clean_up_tokenization_spaces=True)
    
    text = text[:text.find('<endoftext>')].strip()
    if('<sep>' in text):
        text = text[text.find('<sep>') + 5:].strip()
    else:
        text = text[len(prompt + ' <sep>'):].strip()

    return text


import logging
import re
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

import logging
set_global_logging_level(logging.ERROR)


def saveModelBest():
    print("Saving best model")
    os.makedirs(args.save + '/best', exist_ok=True)

    # ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).save_pretrained(args.save + '/best')
    torch.save(ppo_trainer.model.state_dict(), args.save +'/best' +  '/stateDict.pth')
    torch.save(ppo_trainer.model.pretrained_model.state_dict(), args.save + '/best' + '/stateDict_2.pth')

def saveModel():
    print("Saving model")
    os.makedirs(args.save, exist_ok=True)

    # ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).save_pretrained(args.save)
    torch.save(ppo_trainer.model.state_dict(), args.save + '/stateDict.pth')
    torch.save(ppo_trainer.model.pretrained_model.state_dict(), args.save + '/stateDict_2.pth')

num_epochs = 4
from tqdm import tqdm
# from utils.generate import getAccuracy
# best_acc = getAccuracy(ppo_trainer.model, ppo_trainer.tokenizer,'/home/joykirat/JS/GPTJPPO/dataset/svamp.json')
best_acc = 0
print("Training PPO model")
baseReward = 0
max_reward = 0
for epoch in tqdm(range(num_epochs)):
    print("Epoch: ", epoch)
    counter = 0
    for _, data_batch in tqdm(enumerate(ppo_trainer.dataloader)):
        try:
            texts = data_batch['text']
            input_ids = data_batch['input_ids']
            attention_masks = data_batch['attention_mask']
            # print(input_ids)
            # print(sentence)
            response_tensors = []
            response_texts = []
            question_texts = []
            gold_answers = []
            query_tensors = []
            query_texts = []
            gold_response_texts = []
            rewards = []
            pointer = 0
            for query in tqdm(texts):

                sentence = query
                prompt = sentence[: sentence.find('<sep>')].strip()
                target = sentence[sentence.find('<sep>')+5:]
                target = target[: target.find('<endoftext>')]

                XtoYInput = prompt
                query_tensor, response_tensor, response_text = generate(ppo_trainer, tokenizer, XtoYInput, target)
                query_tensors.append(query_tensor.squeeze())
                response_tensors.append(response_tensor.squeeze())
                reward = getReward(response_text, prompt, target, getGoldAnswer(target))
                rewards.append(reward)

                print("reward: ",reward)
                print()

                response_texts.append(response_text)
                query_texts.append(target)
                pointer += 1


            print("rewards: ", rewards)
            reward_sum = 0
            for r in rewards:
                reward_sum += r.item()
            if(max_reward < reward_sum):
                saveModelBest()
                max_reward = reward_sum
            if(counter == 0):
                baseReward = reward_sum
            print("reward sum: ", reward_sum)
            print("Base reward: ", baseReward)
            if(reward_sum > baseReward):
                saveModel()
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, {'query' : [],'response' :[]}, rewards)
            counter += 1
 
        except Exception as e:
            print(e)
            print(sentence)
            print("Error in training")
            print("\n")
            continue
    from utils.generate import getAccuracy
    acc = getAccuracy(ppo_trainer.model, ppo_trainer.tokenizer, '/home/joykirat/JS/GPTJPPO/dataset/svamp.json')
    print("accuracySVAMP: ", acc)
    if(best_acc < acc):
        best_acc = acc
        saveModel()



