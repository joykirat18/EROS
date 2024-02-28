import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import torch
import logging
from tqdm import tqdm
import math
import argparse
import os
# from utils.pseudoCodeDataloader import getDataloader
# from utils.loadModel import loadLoraModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=88888)
parser.add_argument("--model_name", default="gpt2", type=str)
parser.add_argument("--toknizer_name", default="gpt2", type=str)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--num_train_epochs", default=10, type=int)
parser.add_argument("--warmup", default=0.1, type=float)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--input_text_path", default='pseudoCode-Dataset/pseudoCode_csv_full/', type=str)
parser.add_argument("--save", type=str)

args, _ = parser.parse_known_args()

folder = args.input_text_path.split('/')[1]
model_folder = args.model_name.split('/')[-1]
folder = args.save

print("Saving to folder name: ")
print(folder)
train_path = args.input_text_path + 'train.csv'
val_path = args.input_text_path + 'test.csv'

def loadModel(model_path, tokenizer_path):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token=tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, return_dict=True, low_cpu_mem_usage=True)
    args.max_seq_length = tokenizer.model_max_length

    return model, tokenizer

# def loadLoraCheckPoint():
#     tokenizer = AutoTokenizer.from_pretrained('/home/joykirat/JS/FinalCodeBase/script/demo/tokenizer')
#     peft_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#         )
#     model = AutoModelForCausalLM.from_pretrained('/home/joykirat/JS/saved_model/base_gptj', return_dict=True, low_cpu_mem_usage=True)
#     model = get_peft_model(model, peft_config)
#     # print(model)
#     model.resize_token_embeddings(len(tokenizer))
#     model.load_state_dict(torch.load('/home/joykirat/JS/FinalCodeBase/script/demo/stateDict.pth'))

#     return model, tokenizer


model, tokenizer = loadModel(args.model_name, args.toknizer_name)

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

# def create_labels(inputs):
#     labels=[]
#     for ids,attention_mask in zip(inputs['input_ids'],inputs['attention_mask']):
#         label=ids.copy()
#         real_len=sum(attention_mask)
#         padding_len=len(attention_mask)-sum(attention_mask)
#         label[:]=label[:real_len]+[-100]*padding_len
#         labels.append(label)
#     inputs['labels']=labels

# def combinetext(path):

#     combine = []
#     df = pd.read_csv(path)
#     for index, row in df.iterrows():
#         prompt = row['Original_Text']
#         pseudoCode = row['Summary']
#         text = prompt + ' <sep> ' + pseudoCode + ' <|endoftext|>'
#         combine.append(text)

#     return combine

# def create_labels(inputs):
#     labels=[]
#     for ids,attention_mask in zip(inputs['input_ids'],inputs['attention_mask']):
#         label=ids.copy()
#         real_len=sum(attention_mask)
#         padding_len=len(attention_mask)-sum(attention_mask)
#         label[:]=label[:real_len]+[-100]*padding_len
#         labels.append(label)
#     inputs['labels']=labels

# class policySummarizationDataset:
#     def __init__(self, input):
#         self.ids = input['input_ids']
#         self.attention_mask = input['attention_mask']
#         self.labels=input['labels']
        

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, item):
#         return [
#             torch.tensor(self.ids[item], dtype=torch.long),
#             torch.tensor(self.attention_mask[item], dtype=torch.long),
#             torch.tensor(self.labels[item], dtype=torch.long),
#         ]
# from torch.utils.data import Dataset

# def getDataloader(path, tokenizer, args):
#     text = combinetext(path)
#     print("max length tokenizer: ", args.max_seq_length)
#     inputs=tokenizer(text, add_special_tokens=True, padding='max_length',truncation=True,max_length=args.max_seq_length)
#     create_labels(inputs)

#     dataset=policySummarizationDataset(inputs)
#     data_loader = torch.utils.data.DataLoader(
#     dataset,
#     shuffle=False,
#     batch_size=args.batch_size)

#     return data_loader

def getDataset(path):
    import pandas as pd
    from datasets import Dataset
    data = pd.read_csv(path)
    dataset = Dataset.from_pandas(data)

    return dataset

train_dataset = getDataset(train_path)
valid_dataset = getDataset(val_path)

prefix = "summarize: "

# breakpoint()

def preprocess_function(examples):
    inputs = [doc for doc in examples["Original_Text"]]
    model_inputs = tokenizer(inputs, padding=True, max_length=tokenizer.model_max_length, truncation=True)

    labels = tokenizer(examples["Summary"],padding=True, max_length=tokenizer.model_max_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_data = train_dataset.map(preprocess_function, batched=True)
tokenized_valid_data = valid_dataset.map(preprocess_function, batched=True)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.model_name)

import evaluate

rouge = evaluate.load("rouge")

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


training_args = Seq2SeqTrainingArguments(
    output_dir="bartFinetuned2",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=15,
    predict_with_generate=True,
    # report_to="none",
    fp16=True,
    # push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_valid_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
print("Training started")
trainer.train()

trainer.save_model("bartFinetunedModel2")



# tokenized_billsum = billsum.map(preprocess_function, batched=True)

# train_dataloader = getDataloader(train_path, tokenizer, args)
# valid_dataloader = getDataloader(val_path, tokenizer, args)


# num_train_epochs = args.num_train_epochs
# training_steps_per_epoch=len(train_dataloader)
# total_num_training_steps = int(training_steps_per_epoch*num_train_epochs)
# weight_decay=0
# learning_rate=args.learning_rate
# adam_epsilon=1e-6
# warmup_steps=int(total_num_training_steps*args.warmup)
# no_decay = ["bias", "LayerNorm.weight"]
# optimizer_grouped_parameters = [
#     {
#         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#         "weight_decay": weight_decay,
#     },
#     {
#         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#         "weight_decay": 0.0,
#     },
# ]

# optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, eps=adam_epsilon)
# scheduler = get_linear_schedule_with_warmup(
#     optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_num_training_steps
# )

# print("***** Running training *****")
# print("  Total_num_training_step = {}".format(total_num_training_steps))
# print("  Num Epochs = {}".format(num_train_epochs))
# print(f"  Batch_size per device = {args.batch_size}")

# def save_model():
#     print("saving model")
#     model.save_pretrained(folder + '/model')
#     tokenizer.save_pretrained(folder + '/tokenizer')
#     torch.save(model.state_dict(), folder + '/stateDict.pth')

# import os
# os.makedirs(folder, exist_ok=True)
# save_model()
# lowest_loss=10000
# model.to('cuda')
# for epoch in range(num_train_epochs):
#     print(f"Start epoch{epoch+1} of {num_train_epochs}")
#     train_loss=0
#     epoch_iterator = tqdm(train_dataloader,desc='Iteration')
#     model.train()
#     model.zero_grad()    
#     for _, inputs in enumerate(epoch_iterator):        
#         d1,d2,d3=inputs
#         d1=d1.to('cuda')
#         d2=d2.to('cuda')
#         d3=d3.to('cuda')

#         optimizer.zero_grad()
#         output_XtoY = model(input_ids=d1, attention_mask=d2,labels=d3)
#         batch_loss_XtoY=output_XtoY[0]
#         batch_loss=batch_loss_XtoY
#         batch_loss.backward()
#         optimizer.step()
#         scheduler.step()
#         model.zero_grad()
#         train_loss+=batch_loss.item()
#         epoch_iterator.set_description('(batch loss=%g)' % batch_loss.item())
#         del batch_loss
#     print(f'Average train loss per example={train_loss/training_steps_per_epoch} in epoch{epoch+1}')    
#     print(f'Starting evaluate after epoch {epoch+1}')
#     eval_loss=[]    
#     model.eval()    
#     for inputs in tqdm(valid_dataloader, desc="eval"):
#         d1,d2,d3=inputs
#         d1=d1.to('cuda')        
#         d2=d2.to('cuda')
#         d3=d3.to('cuda')

#         with torch.no_grad():
#             output_XtoY = model(input_ids=d1, attention_mask=d2,labels=d3)
#             batch_loss_XtoY=output_XtoY[0]

#             batch_loss=batch_loss_XtoY
#         eval_loss+=[batch_loss.cpu().item()]
#         del batch_loss
#     eval_loss=np.mean(eval_loss)
#     perplexity=math.exp(eval_loss)
#     print(f'Average valid loss per example={eval_loss} in epoch{epoch+1}')    
#     print(f'Perplextiy for valid dataset in epoch{epoch+1} is {perplexity}')
#     if(eval_loss < lowest_loss):
#         lowest_loss = eval_loss
#         save_model()





