from tqdm import tqdm
from torch import nn
import pickle
# from lion_pytorch import Lion
import argparse
import time
import os
# from collections import namedtuple
from typing import Dict
from collections import namedtuple
# import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.optim import SGD

from dataloaders.dataload import BERTNERDataset
from dataloaders.truncate_dataset import TruncateDataset
from dataloaders.collate_functions import collate_to_max_length
from models.bert_model_spanner import BertNER
from models.config_spanner import BertNerConfig
# from utils.get_parser import get_parser
from radom_seed import set_random_seed
from eval_metric import span_f1, span_f1_prune, get_predict, get_predict_prune
import random
import logging
logger = logging.getLogger(__name__)
set_random_seed(0)


from trainer import BertNerTagger, get_train_dataloader, get_val_dataloader, get_test_dataloader, configure_optimizers


def main():
    """main"""
    # parser = get_parser()

    # add model specific args
    parser = BertNerTagger.get_parser()

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    # args.device = torch.device(args.device)

    # begin{add label2indx augument into the args.}
    label2idx = {}
    if 'conll' in args.dataname:
        label2idx = {"O": 0, "ORG": 1, "PER": 2, "LOC": 3, "MISC": 4}
    elif 'note' in args.dataname:
        label2idx = {'O': 0, 'PERSON': 1, 'ORG': 2, 'GPE': 3, 'DATE': 4, 'NORP': 5, 'CARDINAL': 6, 'TIME': 7,
                     'LOC': 8,
                     'FAC': 9, 'PRODUCT': 10, 'WORK_OF_ART': 11, 'MONEY': 12, 'ORDINAL': 13, 'QUANTITY': 14,
                     'EVENT': 15,
                     'PERCENT': 16, 'LAW': 17, 'LANGUAGE': 18}
    elif args.dataname == 'wnut16':
        label2idx = {'O': 0, 'loc': 1, 'facility': 2, 'movie': 3, 'company': 4, 'product': 5, 'person': 6, 'other': 7,
                     'tvshow': 8, 'musicartist': 9, 'sportsteam': 10}
    elif args.dataname == 'wnut17':
        label2idx = {'O': 0, 'location': 1, 'group': 2, 'corporation': 3,
                     'person': 4, 'creative-work': 5, 'product': 6}
    elif args.dataname == 'fepd':
        label2idx = {'O': 0, 'target_direct': 1, 'source_direct': 2, 'source_indirect': 3, 'data': 4,
                     'reason': 5, 'data_compulsory': 6, 'medium': 7, 'target_in_direct': 8, 'data_optional': 9}

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
    # print(model)
    model = model.to(args.device)
    state_dict = torch.load('/home/rohan19095/BTP/SpanNer-Final/spanner_bert-large-uncased_spMLen_usePruneFalse_useSpLenTrue_useSpMorphFalse_SpWtFalse_value0.5best_model_dev.pth',map_location='cuda:0')

    model.load_state_dict(state_dict)
    # breakpoint()
    print("model loaded")
    train_dataloader = get_train_dataloader(args)
    val_dataloader = get_val_dataloader(args)
    test_dataloader = get_test_dataloader(args)

    # OPTIMIZER
    optimizers, schedulers = configure_optimizers(model, args)
    model.eval()
    outputs = []
    with torch.no_grad():
        val_loss = []
        print("Test starting!")
        for batch in tqdm(test_dataloader):

            output = {}

            # tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
            tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = batch
            loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken,
                       all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs]

            attention_mask = (tokens != 0).long()
            all_span_rep, all_span_feature_rep = model.forward(
                loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids)
            predicts = model.classifier(all_span_rep)

            # pred_label_idx_new = torch.zeros_like(real_span_mask_ltoken)
            predicts = predicts.to(args.device)
            span_label_ltoken = span_label_ltoken.to(args.device)
            real_span_mask_ltoken = real_span_mask_ltoken.to(args.device)
            span_f1s = span_f1(
                predicts, span_label_ltoken, real_span_mask_ltoken)
            batch_preds = get_predict(args, all_span_word, words, predicts, span_label_ltoken,
                                        all_span_idxs)
            output["span_f1s"] = span_f1s

            output["batch_preds"] = batch_preds

            output["predicts"] = predicts
            output['all_span_word'] = all_span_word
            outputs.append(output)

        print("use... test_epoch_end: ",)

        all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
        correct_pred, total_pred, total_golden = all_counts
        print('correct_pred, total_pred, total_golden: ',
              correct_pred, total_pred, total_golden)
        precision = correct_pred / (total_pred + 1e-10)
        recall = correct_pred / (total_golden + 1e-10)
        f1 = precision * recall * 2 / (precision + recall + 1e-10)

        print("span_precision: ", precision)
        print("span_recall: ", recall)
        print("span_f1: ", f1)
        # tensorboard_logs[f"span_precision"] = precision
        # tensorboard_logs[f"span_recall"] = recall
        # tensorboard_logs[f"span_f1"] = f1

        # begin{save the predict results}
        pred_batch_results = [x['batch_preds'] for x in outputs]
        fp_write = args.default_root_dir + '/'+args.modelName + '_test.txt'
        fwrite = open(fp_write, 'w')
        for pred_batch_result in pred_batch_results:
            for pred_result in pred_batch_result:
                # print("pred_result: ", pred_result)
                fwrite.write(pred_result+'\n')

        print("test: %f, %f, %f, %d, %d, %d\n" %
              (f1, recall, precision, correct_pred, total_pred, total_golden))
        # end{save the predict results}

        # begin{save the predict prob}
        all_predicts = [list(x['predicts'].cpu()) for x in outputs]
        all_span_words = [list(x['all_span_word']) for x in outputs]

        # begin{get the label2idx dictionary}
        label2idx = {}
        label2idx_list = args.label2idx_list
        for labidx in label2idx_list:
            lab, idx = labidx
            label2idx[lab] = int(idx)
            # end{get the label2idx dictionary}

        file_prob1 = args.default_root_dir + '/'+args.modelName + '_prob_test.pkl'
        print("the file path of probs: ", file_prob1)
        fwrite_prob = open(file_prob1, 'wb')
        pickle.dump([label2idx, all_predicts, all_span_words], fwrite_prob)
        # end{save the predict prob...}

        # print("Test: ", {'val_loss': avg_loss, 'log': tensorboard_logs})


if __name__ == '__main__':
    # run_dataloader()
    main()