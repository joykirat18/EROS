# encoding: utf-8


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


class BertNerTagger(nn.Module):
    """MLM Trainer"""

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        if isinstance(args, argparse.Namespace):
            # self.save_hyperparameters(args)
            self.args = args
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)
            # print(self.args)
            # print(args)

        self.bert_dir = args.bert_config_dir
        # self.data_dir = self.args.data_dir

        bert_config = BertNerConfig.from_pretrained(args.bert_config_dir,
                                                    hidden_dropout_prob=args.bert_dropout,
                                                    attention_probs_dropout_prob=args.bert_dropout,
                                                    model_dropout=args.model_dropout)

        self.model = BertNER.from_pretrained(args.bert_config_dir,
                                             config=bert_config,
                                             args=self.args)
        self.model = self.model.to(self.args.device)
        # print(self.model)
        logging.info(str(args.__dict__ if isinstance(
            args, argparse.ArgumentParser) else args))
        # self.results = []
        # self.optimizer = args.optimizer
        self.n_class = args.n_class

        self.max_spanLen = args.max_spanLen
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.classifier = torch.nn.Softmax(dim=-1)
        self.lossRatio = 0.7

        self.fwrite_epoch_res = open(args.fp_epoch_result, 'w')
        self.fwrite_epoch_res.write(
            "f1, recall, precision, correct_pred, total_pred, total_golden\n")
        # print("WRITTEN TO EPOCH RESULTS")

    @staticmethod
    def get_parser():
        def str2bool(v):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')

        parser = argparse.ArgumentParser(description="Training")

        # basic argument&value
        parser.add_argument("--data_dir", type=str,
                            required=True, help="data dir")
        parser.add_argument("--bert_config_dir", type=str,
                            required=True, help="bert config dir")
        parser.add_argument("--pretrained_checkpoint", default="",
                            type=str, help="pretrained checkpoint path")
        parser.add_argument("--bert_max_length", type=int,
                            default=128, help="max length of dataset")
        parser.add_argument("--batch_size", type=int,
                            default=10, help="batch size")
        parser.add_argument("--lr", type=float,
                            default=1e-5, help="learning rate")
        parser.add_argument("--workers", type=int, default=32,
                            help="num workers for dataloader")
        parser.add_argument("--weight_decay", default=0.01, type=float,
                            help="Weight decay if we apply some.")
        parser.add_argument("--warmup_steps", default=0, type=int,
                            help="warmup steps used for scheduler.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")

        parser.add_argument("--device", type=str, default="cpu", help="device")
        parser.add_argument("--model_dropout", type=float, default=0.2,
                            help="model dropout rate")
        parser.add_argument("--bert_dropout", type=float, default=0.2,
                            help="bert dropout rate")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        parser.add_argument("--optimizer", choices=["adamw", "sgd", "lion"], default="lion",
                            help="loss type")
        # choices=["conll03", "ace04","notebn","notebc","notewb","notemz",'notenw','notetc']
        parser.add_argument("--dataname", default="conll03",
                            help="the name of a dataset")
        parser.add_argument("--max_spanLen", type=int,
                            default=4, help="max span length")
        # parser.add_argument("--margin", type=float, default=0.03, help="margin of the ranking loss")
        parser.add_argument("--n_class", type=int, default=5,
                            help="the classes of a task")
        parser.add_argument("--modelName",  default='test',
                            help="the classes of a task")

        # parser.add_argument('--use_allspan', type=str2bool, default=True, help='use all the spans with O-labels ', nargs='?',
        #                     choices=['yes (default)', True, 'no', False])

        parser.add_argument('--use_tokenLen', type=str2bool, default=False, help='use the token length (after the bert tokenizer process) as a feature',
                            nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--tokenLen_emb_dim", type=int,
                            default=50, help="the embedding dim of a span")
        parser.add_argument('--span_combination_mode', default='x,y',
                            help='Train data in format defined by --data-io param.')

        parser.add_argument('--use_spanLen', type=str2bool, default=False, help='use the span length as a feature',
                            nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--spanLen_emb_dim",  type=int,
                            default=100, help="the embedding dim of a span length")

        parser.add_argument('--use_morph', type=str2bool, default=True, help='use the span length as a feature',
                            nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--morph_emb_dim", type=int, default=100,
                            help="the embedding dim of the morphology feature.")
        parser.add_argument('--morph2idx_list', type=list,
                            help='a list to store a pair of (morph, index).', )

        parser.add_argument('--label2idx_list', type=list,
                            help='a list to store a pair of (label, index).',)

        random_int = '%08d' % (random.randint(0, 100000000))
        # print('random_int:', random_int)

        parser.add_argument('--random_int', type=str, default=random_int,
                            help='a list to store a pair of (label, index).', )
        parser.add_argument('--param_name', type=str, default='param_name',
                            help='a prexfix for a param file name', )
        parser.add_argument('--best_dev_f1', type=float, default=0.0,
                            help='best_dev_f1 value', )
        parser.add_argument('--use_prune', type=str2bool, default=True,
                            help='best_dev_f1 value', )

        parser.add_argument("--use_span_weight", type=str2bool, default=True,
                            help="range: [0,1.0], the weight of negative span for the loss.")
        parser.add_argument("--neg_span_weight", type=float, default=0.5,
                            help="range: [0,1.0], the weight of negative span for the loss.")
        return parser

    def forward(self, loadall, all_span_lens, all_span_idxs_ltoken, input_ids, attention_mask, token_type_ids):
        """"""
        return self.model(loadall, all_span_lens, all_span_idxs_ltoken, input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    # def training_step(self, batch, batch_idx):
    #     """"""
    #     tf_board_logs = {
    #         "lr": self.trainer.optimizers[0].param_groups[0]['lr']
    #     }
    #     # tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
    #     # print(1)
    #     tokens, token_type_ids, all_span_idxs_ltoken,morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,real_span_mask_ltoken,words,all_span_word,all_span_idxs = batch
    #     loadall = [tokens, token_type_ids, all_span_idxs_ltoken,morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,
    #                real_span_mask_ltoken, words, all_span_word, all_span_idxs]

    #     attention_mask = (tokens != 0).long()
    #     all_span_rep, all_span_feature_rep = self.forward(loadall,all_span_lens,all_span_idxs_ltoken,tokens, attention_mask, token_type_ids)
    #     # print(2)
    #     predicts = self.classifier(all_span_rep)
    #     # print('all_span_rep.shape: ', all_span_rep.shape)
    #     # print(3)
    #     output = {}
    #     if self.args.use_prune:
    #         span_f1s,pred_label_idx = span_f1_prune(all_span_idxs, predicts, span_label_ltoken, real_span_mask_ltoken)
    #     else:
    #         span_f1s = span_f1(predicts, span_label_ltoken, real_span_mask_ltoken)
    #     output["span_f1s"] = span_f1s
    #     # print(4)
    #     loss = self.compute_loss(loadall,all_span_rep, span_label_ltoken, real_span_mask_ltoken,all_span_feature_rep,mode='train')
    #     # print(5)
        # contrastive_loss = self.compute_CL_loss(loadall,all_span_rep, span_label_ltoken, real_span_mask_ltoken,all_span_feature_rep,mode='train')
    #     # print(6)
    #     # print(,contrastive_loss)
    #     final_loss = self.lossRatio * loss + (1 - self.lossRatio) * contrastive_loss
    #     print(final_loss.item(), loss.item(), contrastive_loss.item())
    #     output[f"train_loss"] = final_loss

    #     tf_board_logs[f"loss"] = final_loss
    #     tf_board_logs[f"Closs"] = loss
    #     tf_board_logs[f"contrastive"] = contrastive_loss
    #     output['loss'] = final_loss
    #     output['log'] =tf_board_logs
    #     # print(7)
    #     return output

    # def training_epoch_end(self, outputs):
    #     """"""
    #     print("use... training_epoch_end: ", )
    #     avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'train_loss': avg_loss}
    #     all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
    #     correct_pred, total_pred, total_golden = all_counts
    #     print('in train correct_pred, total_pred, total_golden: ', correct_pred, total_pred, total_golden)
    #     precision =correct_pred / (total_pred+1e-10)
    #     recall = correct_pred / (total_golden + 1e-10)
    #     f1 = precision * recall * 2 / (precision + recall + 1e-10)

    #     print("in train span_precision: ", precision)
    #     print("in train span_recall: ", recall)
    #     print("in train span_f1: ", f1)
    #     tensorboard_logs[f"span_precision"] = precision
    #     tensorboard_logs[f"span_recall"] = recall
    #     tensorboard_logs[f"span_f1"] = f1

    #     self.fwrite_epoch_res.write(
    #         "train: %f, %f, %f, %d, %d, %d\n" % (f1, recall, precision, correct_pred, total_pred, total_golden))

    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # def validation_step(self, batch, batch_idx):
    #     """"""

    #     output = {}

    #     # tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
    #     tokens, token_type_ids, all_span_idxs_ltoken,morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,real_span_mask_ltoken,words,all_span_word,all_span_idxs = batch
    #     loadall = [tokens, token_type_ids, all_span_idxs_ltoken,morph_idxs, span_label_ltoken, all_span_lens,all_span_weights,real_span_mask_ltoken,words,all_span_word,all_span_idxs]

    #     attention_mask = (tokens != 0).long()
    #     all_span_rep, all_span_feature_rep = self.forward(loadall,all_span_lens,all_span_idxs_ltoken, tokens, attention_mask, token_type_ids)
    #     predicts = self.classifier(all_span_rep)

    #     # pred_label_idx_new = torch.zeros_like(real_span_mask_ltoken)
    #     if self.args.use_prune:
    #         span_f1s,pred_label_idx = span_f1_prune(all_span_idxs, predicts, span_label_ltoken, real_span_mask_ltoken)
    #         # print('pred_label_idx_new: ',pred_label_idx_new.shape)
    #         # print('predicts: ', predicts.shape)
    #         # print('pred_label_idx_new: ',pred_label_idx_new)
    #         # print('predicts: ', predicts)

    #         batch_preds = get_predict_prune(self.args, all_span_word, words, pred_label_idx, span_label_ltoken,
    #                                            all_span_idxs)
    #     else:
    #         span_f1s = span_f1(predicts, span_label_ltoken, real_span_mask_ltoken)
    #         batch_preds = get_predict(self.args, all_span_word, words, predicts, span_label_ltoken,
    #                                            all_span_idxs)
    #     self.results.append([predicts,loadall,all_span_rep,all_span_feature_rep, span_label_ltoken, real_span_mask_ltoken,all_span_word, words, all_span_idxs])
    #     fwrite_prob = open('results5.pickle', 'wb')
    #     pickle.dump(self.results, fwrite_prob)
    #     output["span_f1s"] = span_f1s
    #     loss = self.compute_loss(loadall,all_span_rep, span_label_ltoken, real_span_mask_ltoken,all_span_feature_rep,mode='test/dev')
    #     # print("Feature rep ", all_span_feature_rep.shape)
    #     contrastive_loss = self.compute_CL_loss(loadall,all_span_rep, span_label_ltoken, real_span_mask_ltoken,all_span_feature_rep,mode='test/val')

    #     output["batch_preds"] =batch_preds
    #     final_loss = self.lossRatio * loss + (1 - self.lossRatio) * contrastive_loss
    #     # output["batch_preds_prune"] = pred_label_idx_new
    #     output[f"val_loss"] = final_loss

    #     output["predicts"] = predicts
    #     output['all_span_word'] = all_span_word

    #     return output

    # def validation_epoch_end(self, outputs):
    #     """"""
    #     print("use... validation_epoch_end: ", )
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
    #     correct_pred, total_pred, total_golden = all_counts
    #     print('correct_pred, total_pred, total_golden: ', correct_pred, total_pred, total_golden)
    #     precision =correct_pred / (total_pred+1e-10)
    #     recall = correct_pred / (total_golden + 1e-10)
    #     f1 = precision * recall * 2 / (precision + recall + 1e-10)

    #     print("span_precision: ", precision)
    #     print("span_recall: ", recall)
    #     print("span_f1: ", f1)
    #     tensorboard_logs[f"span_precision"] = precision
    #     tensorboard_logs[f"span_recall"] = recall
    #     tensorboard_logs[f"span_f1"] = f1
    #     # with open('epoch_results2','w') as f:
    #     #     f.write
    #     self.fwrite_epoch_res.write("dev: %f, %f, %f, %d, %d, %d\n"%(f1,recall,precision,correct_pred, total_pred, total_golden))

    #     # print("Written epoch results")

    #     if f1>self.args.best_dev_f1:
    #         pred_batch_results = [x['batch_preds'] for x in outputs]
    #         fp_write = self.args.default_root_dir +  '/' + self.args.modelName + '_dev.txt'
    #         fwrite = open(fp_write, 'w')
    #         for pred_batch_result in pred_batch_results:
    #             for pred_result in pred_batch_result:
    #                 # print("pred_result: ", pred_result)
    #                 fwrite.write(pred_result + '\n')
    #         self.args.best_dev_f1=f1

    #         # begin{save the predict prob}
    #         all_predicts = [list(x['predicts']) for x in outputs]
    #         all_span_words = [list(x['all_span_word']) for x in outputs]

    #         # begin{get the label2idx dictionary}
    #         label2idx = {}
    #         label2idx_list = self.args.label2idx_list
    #         for labidx in label2idx_list:
    #             lab, idx = labidx
    #             label2idx[lab] = int(idx)
    #             # end{get the label2idx dictionary}

    #         file_prob1 = self.args.default_root_dir + '/' + self.args.modelName + '_prob_dev.pkl'
    #         print("the file path of probs: ", file_prob1)
    #         fwrite_prob = open(file_prob1, 'wb')
    #         pickle.dump([label2idx, all_predicts, all_span_words], fwrite_prob)
    #         # end{save the predict prob...}

    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """"""
        # print("use... test_step: ",)
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(
        self,
        outputs
    ) -> Dict[str, Dict[str, Tensor]]:
        """"""
        print("use... test_epoch_end: ",)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
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
        tensorboard_logs[f"span_precision"] = precision
        tensorboard_logs[f"span_recall"] = recall
        tensorboard_logs[f"span_f1"] = f1

        # begin{save the predict results}
        pred_batch_results = [x['batch_preds'] for x in outputs]
        fp_write = self.args.default_root_dir + '/'+self.args.modelName + '_test.txt'
        fwrite = open(fp_write, 'w')
        for pred_batch_result in pred_batch_results:
            for pred_result in pred_batch_result:
                # print("pred_result: ", pred_result)
                fwrite.write(pred_result+'\n')

        self.fwrite_epoch_res.write(
            "test: %f, %f, %f, %d, %d, %d\n" % (f1, recall, precision, correct_pred, total_pred, total_golden))
        # end{save the predict results}

        # begin{save the predict prob}
        all_predicts = [list(x['predicts'].cpu()) for x in outputs]
        all_span_words = [list(x['all_span_word']) for x in outputs]

        # begin{get the label2idx dictionary}
        label2idx = {}
        label2idx_list = self.args.label2idx_list
        for labidx in label2idx_list:
            lab, idx = labidx
            label2idx[lab] = int(idx)
            # end{get the label2idx dictionary}

        file_prob1 = self.args.default_root_dir + \
            '/'+self.args.modelName + '_prob_test.pkl'
        print("the file path of probs: ", file_prob1)
        fwrite_prob = open(file_prob1, 'wb')
        pickle.dump([label2idx, all_predicts, all_span_words], fwrite_prob)
        # end{save the predict prob...}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}


def compute_loss(model, loadall, all_span_rep, span_label_ltoken, real_span_mask_ltoken, all_span_feature_rep, mode):
    '''

    :param all_span_rep: shape: (bs, n_span, n_class)
    :param span_label_ltoken:
    :param real_span_mask_ltoken:
    :return:
    '''
    batch_size, n_span = span_label_ltoken.size()
    all_span_rep1 = all_span_rep.view(-1, model.n_class)
    span_label_ltoken1 = span_label_ltoken.view(-1)
    loss = model.cross_entropy(all_span_rep1, span_label_ltoken1)
    loss = loss.view(batch_size, n_span)
    # print('loss 1: ', loss)
    if mode == 'train' and model.args.use_span_weight:  # when training we should multiply the span-weight
        span_weight = loadall[6]
        loss = loss*span_weight
        # print('loss 2: ', loss)

    loss = torch.masked_select(loss, real_span_mask_ltoken.bool())

    # print("1 loss: ", loss)
    loss = torch.mean(loss)
    # print("loss: ", loss)
    predict = model.classifier(all_span_rep)  # shape: (bs, n_span, n_class)

    return loss


def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = torch.divide(a, torch.max(a_n, eps * torch.ones_like(a_n)))
    b_norm = torch.divide(b, torch.max(b_n, eps * torch.ones_like(b_n)))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def compute_CL_loss(args, loadall, all_span_rep, span_label_ltoken, real_span_mask_ltoken, all_span_feature_rep, mode):
    batch_size, n_span = span_label_ltoken.size()

    mask_ = real_span_mask_ltoken.unsqueeze(
        -1).expand(all_span_feature_rep.size())
    embedding = torch.masked_select(
        all_span_feature_rep, (mask_ == 1)).view(-1, 512)
    label = torch.masked_select(
        span_label_ltoken, real_span_mask_ltoken.bool())
    cosine_sim = sim_matrix(embedding, embedding)
    dis = cosine_sim[~torch.eye(cosine_sim.shape[0], dtype=torch.bool)].reshape(
        cosine_sim.shape[0], -1)
    temp = 0.1
    scale = 1000

    dis = dis / temp
    cosine_sim = cosine_sim / temp

    dis = torch.exp(dis)
    cosine_sim = torch.exp(cosine_sim)

    row_sum = torch.sum(dis, dim=1)

    contrastive_loss = 0
    from tqdm import tqdm
    for i in range(len(embedding)):
        n_i = (label == label[i]).sum() - 1

        if n_i != 0:
            sameEmbedding = torch.nonzero(label == label[i]).squeeze()

            inner_sum = torch.sum(
                torch.log(cosine_sim[i, sameEmbedding] / row_sum[i]))
            inner_sum -= torch.log(cosine_sim[i, i] / row_sum[i])
            # inner_sum -= torch.log(cosine_sim[i, i] / row_sum[i])

            contrastive_loss += inner_sum / (-n_i)

    return contrastive_loss / scale




def get_dataloader(args, prefix="train", limit: int = None) -> DataLoader:
    """get training dataloader"""
    """
    load_mmap_dataset
    """
    json_path = os.path.join(args.data_dir, f"spanner.{prefix}")
    # print("json_path: ", json_path)
    # vocab_path = os.path.join(self.bert_dir, "vocab.txt")
    # dataset = BERTNERDataset(self.args,json_path=json_path,
    #                         tokenizer=BertWordPieceTokenizer(vocab_path),
    #                         # tokenizer=BertWordPieceTokenizer(vocab_file=vocab_path),
    #                         max_length=self.args.bert_max_length,
    #                         pad_to_maxlen=False
    #                         )

    # vocab_path = os.path.join(self.bert_dir, "vocab.txt")
    # print("use BertWordPieceTokenizer as the tokenizer ")
    import json
    all_data = json.load(open(json_path, encoding="utf-8"))
    dataset = BERTNERDataset(args, all_data=all_data,
                             tokenizer=BertWordPieceTokenizer("vocab.txt"),
                             # tokenizer=BertWordPieceTokenizer(vocab_file=vocab_path),
                             max_length=args.bert_max_length,
                             pad_to_maxlen=False
                             )

    if limit is not None:
        dataset = TruncateDataset(dataset, limit)
    # dataset.to(args.device
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        # shuffle=True if prefix == "train" else False,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_to_max_length
    )
    return dataloader

def get_customDataloader(args,sentence, prefix="train", limit: int = None) -> DataLoader:
    """get training dataloader"""
    """
    load_mmap_dataset
    """
    json_path = os.path.join(args.data_dir, f"spanner.{prefix}")
    # print("json_path: ", json_path)

    # print("use BertWordPieceTokenizer as the tokenizer ")
    sentences = sentence.split('. ')
    all_data = []
    for s in sentences:
        if(len(s.strip()) != 0):
            all_data.append({'context': s.strip(), 'span_posLabel': {}})
            # print(s.strip())

    dataset = BERTNERDataset(args, all_data=all_data,
                             tokenizer=BertWordPieceTokenizer("vocab.txt"),
                             # tokenizer=BertWordPieceTokenizer(vocab_file=vocab_path),
                             max_length=args.bert_max_length,
                             pad_to_maxlen=False
                             )

    if limit is not None:
        dataset = TruncateDataset(dataset, limit)
    # dataset.to(args.device
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        # shuffle=True if prefix == "train" else False,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_to_max_length
    )
    return dataloader


def get_train_dataloader(args) -> DataLoader:
    return get_dataloader(args, "train")
    # return self.get_dataloader("dev", 100)


def get_val_dataloader(args):
    val_data = get_dataloader(args, "dev")
    return val_data


def get_test_dataloader(args):
    return get_dataloader(args, "test")

def get_Custom_dataloader(args, sentence):
    return get_customDataloader(args,sentence, "test")
    # return self.get_dataloader("dev")


def configure_optimizers(model, args):
    """Prepare optimizer and schedule (linear warmup and decay)"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.optimizer == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=args.lr,
                          eps=args.adam_epsilon,)
    elif args.optimizer == "lion":
        # typically asked to keep 1/10th of AdamW opt
        optimizer = Lion(optimizer_grouped_parameters, lr=args.lr)
    else:
        optimizer = SGD(optimizer_grouped_parameters, lr=args.lr, momentum=0.9)

    num_gpus = len([x for x in str(args.gpus).split(",") if x.strip()])
    t_total = (len(get_train_dataloader(args)) // (args.accumulate_grad_batches *
               num_gpus) + 1) * args.max_epochs  # WARN: POTENTIAL PROBLEM
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, pct_start=float(args.warmup_steps/t_total),
        final_div_factor=args.final_div_factor,
        total_steps=t_total, anneal_strategy='linear'
    )
    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


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
    if args.pretrained_checkpoint:
        model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                         map_location=torch.device('cpu'))["state_dict"])

    # # save the best model
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=args.default_root_dir,
    #     save_top_k=1,
    #     verbose=True,
    #     monitor="span_f1",
    #     # period=-1,
    #     mode="max",
    # )

    # DATA LOADERS
    train_dataloader = get_train_dataloader(args)
    val_dataloader = get_val_dataloader(args)
    test_dataloader = get_test_dataloader(args)

    # OPTIMIZER
    optimizers, schedulers = configure_optimizers(model, args)

    # TRAINING LOOP

    num_epochs = args.max_epochs
    results = []
    for epoch in range(num_epochs):
        print('Epoch: ', epoch)
        # TRAIN LOOP
        outputs = []
        embedding_sizes = []
        model.train()
        count = 0
        # tqdm progress bar for train_dataloader
        for batch in tqdm(train_dataloader):

            tf_board_logs = {
                "lr": optimizers[0].param_groups[0]['lr']
            }

            tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = batch
            loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights,
                       real_span_mask_ltoken, words, all_span_word, all_span_idxs]

            attention_mask = (tokens != 0).long()

            all_span_rep, all_span_feature_rep = model.forward(
                loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids)

            predicts = model.classifier(all_span_rep)

            output = {}
            if args.use_prune:
                span_f1s, pred_label_idx = span_f1_prune(
                    all_span_idxs, predicts, span_label_ltoken, real_span_mask_ltoken)
            else:
                predicts = predicts.to(args.device)
                span_label_ltoken = span_label_ltoken.to(args.device)
                real_span_mask_ltoken = real_span_mask_ltoken.to(args.device)
                span_f1s = span_f1(
                    predicts, span_label_ltoken, real_span_mask_ltoken)
            output["span_f1s"] = span_f1s
            loss = compute_loss(model, loadall, all_span_rep, span_label_ltoken,
                                real_span_mask_ltoken, all_span_feature_rep, mode='train')


            # time start
            # start_time = time.time()
            # args.device = 'cpu'
            all_span_rep = all_span_rep.to(args.device)
            span_label_ltoken = span_label_ltoken.to(args.device)
            real_span_mask_ltoken = real_span_mask_ltoken.to(args.device)
            all_span_feature_rep = all_span_feature_rep.to(args.device)

            # print(5)
            contrastive_loss = compute_CL_loss(
                args, loadall, all_span_rep, span_label_ltoken, real_span_mask_ltoken, all_span_feature_rep, mode='train')
            # contrastive_loss2 = compute_CL_loss_new(args, loadall,all_span_rep, span_label_ltoken, real_span_mask_ltoken,all_span_feature_rep,mode='train')
            # print(f"contrastive loss {loss}, new constrastive loss {contrastive_loss2}")
            # print(contrastive_loss, contrastive_loss2)
            # print(contrastive_loss, unoptimized_cl, contrastive_loss == unoptimized_cl)
            mask_ = real_span_mask_ltoken.unsqueeze(-1).expand(all_span_feature_rep.size())
            embedding = torch.masked_select(all_span_feature_rep, (mask_ == 1)).view(-1, 512)
            embedding_sizes.append(embedding.size(0))

            args.device = 'cuda:0'
            all_span_rep = all_span_rep.to(args.device)
            span_label_ltoken = span_label_ltoken.to(args.device)
            real_span_mask_ltoken = real_span_mask_ltoken.to(args.device)
            all_span_feature_rep = all_span_feature_rep.to(args.device)
            contrastive_loss = torch.tensor(0)
            # print(6)
            # print(,contrastive_loss)
            final_loss = torch.add(torch.multiply(model.lossRatio, loss), torch.multiply(
                (1 - model.lossRatio), contrastive_loss))
            loss = final_loss
            # end_time = time.time()
            print(f"Losses computed: final_loss {final_loss.item()}, CE Loss: {loss.item()}, Contrastive: {contrastive_loss.item()} in time: {end_time - start_time}")
            # print(
            #     f"Losses computed: final_loss {final_loss.item()}, CE Loss: {loss.item()} in time: {end_time - start_time}")
            output[f"train_loss"] = loss

            tf_board_logs[f"loss"] = loss
            # tf_board_logs[f"Closs"] = loss
            tf_board_logs[f"contrastive"] = contrastive_loss
            output['loss'] = loss
            output['log'] = tf_board_logs
            outputs.append(output)
            # start_time = time.time()
            loss.backward()
            # end_time = time.time()
            # print(f"Backward done in time: {end_time - start_time}")
            # start_time = time.time()
            optimizers[0].step()
            # end_time = time.time()
            # print(f"Step done in time: {end_time - start_time}")
            # start_time = time.time()
            optimizers[0].zero_grad()
            # end_time = time.time()
            # print(f"Zero_grad: {end_time - start_time}")

        print("use... training_epoch_end: ", )
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss}
        all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
        correct_pred, total_pred, total_golden = all_counts
        print('in train correct_pred, total_pred, total_golden: ',
              correct_pred, total_pred, total_golden)
        precision = correct_pred / (total_pred+1e-10)
        recall = correct_pred / (total_golden + 1e-10)
        f1 = precision * recall * 2 / (precision + recall + 1e-10)

        print("in train span_precision: ", precision)
        print("in train span_recall: ", recall)
        print("in train span_f1: ", f1)
        tensorboard_logs[f"span_precision"] = precision
        tensorboard_logs[f"span_recall"] = recall
        tensorboard_logs[f"span_f1"] = f1

        print(
            "train: %f, %f, %f, %d, %d, %d\n" % (f1, recall, precision, correct_pred, total_pred, total_golden))

        print({'val_loss': avg_loss, 'log': tensorboard_logs})

        # VALIDATION LOOP
        outputs = []
        model.eval()
        with torch.no_grad():
            val_loss = []
            print("Validation Loop starting!")
            for batch in tqdm(val_dataloader):

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
                if args.use_prune:
                    span_f1s, pred_label_idx = span_f1_prune(
                        all_span_idxs, predicts, span_label_ltoken, real_span_mask_ltoken)
                    # print('pred_label_idx_new: ',pred_label_idx_new.shape)
                    # print('predicts: ', predicts.shape)
                    # print('pred_label_idx_new: ',pred_label_idx_new)
                    # print('predicts: ', predicts)

                    batch_preds = get_predict_prune(args, all_span_word, words, pred_label_idx, span_label_ltoken,
                                                    all_span_idxs)
                else:
                    predicts = predicts.to(args.device)
                    span_label_ltoken = span_label_ltoken.to(args.device)
                    real_span_mask_ltoken = real_span_mask_ltoken.to(
                        args.device)

                    # all_span_idxs = all_span_idxs.to(args.device)

                    span_f1s = span_f1(
                        predicts, span_label_ltoken, real_span_mask_ltoken)
                    # batch_preds = get_predict(args, all_span_word, words, predicts, span_label_ltoken,
                    #   all_span_idxs)
                loss = compute_loss(model, loadall, all_span_rep, span_label_ltoken,
                                    real_span_mask_ltoken, all_span_feature_rep, mode='test/dev')
                output["span_f1s"] = span_f1s
                # args.device = 'cpu'
                # all_span_rep = all_span_rep.to(args.device)
                # span_label_ltoken = span_label_ltoken.to(args.device)
                # real_span_mask_ltoken = real_span_mask_ltoken.to(args.device)
                # all_span_feature_rep = all_span_feature_rep.to(args.device)

                # print("Feature rep ", all_span_feature_rep.shape)
                # contrastive_loss = compute_CL_loss(
                    # args, loadall, all_span_rep, span_label_ltoken, real_span_mask_ltoken, all_span_feature_rep, mode='test/val')
                # contrastive_loss = torch.tensor(0)
                # args.device = 'cuda:0'
                # all_span_rep = all_span_rep.to(args.device)
                # span_label_ltoken = span_label_ltoken.to(args.device)
                # real_span_mask_ltoken = real_span_mask_ltoken.to(args.device)
                # all_span_feature_rep = all_span_feature_rep.to(args.device)

                # output["batch_preds"] = batch_preds
                # final_loss = model.lossRatio * loss + \
                    # (1 - model.lossRatio) * contrastive_loss
                # final_loss = loss
                # output["batch_preds_prune"] = pred_label_idx_new
                output[f"val_loss"] = loss

                output["predicts"] = predicts
                output['all_span_word'] = all_span_word
                outputs.append(output)
                val_loss.append(loss.item())

            val_loss = torch.mean(torch.tensor(val_loss))
            print(f"Validation Loss: {val_loss.item()}")

            print("use... validation_epoch_end: ", )
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            tensorboard_logs = {'val_loss': avg_loss}
            all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
            correct_pred, total_pred, total_golden = all_counts
            print('correct_pred, total_pred, total_golden: ',
                  correct_pred, total_pred, total_golden)
            precision = correct_pred / (total_pred+1e-10)
            recall = correct_pred / (total_golden + 1e-10)
            f1 = precision * recall * 2 / (precision + recall + 1e-10)

            print("span_precision: ", precision)
            print("span_recall: ", recall)
            print("span_f1: ", f1)
            tensorboard_logs[f"span_precision"] = precision
            tensorboard_logs[f"span_recall"] = recall
            tensorboard_logs[f"span_f1"] = f1
            # with open('epoch_results2','w') as f:
            #     f.write
            print("dev: %f, %f, %f, %d, %d, %d\n" %
                  (f1, recall, precision, correct_pred, total_pred, total_golden))

            # print("Written epoch results")

            if f1 > args.best_dev_f1:
                # pred_batch_results = [x['batch_preds'] for x in outputs]
                # fp_write = args.default_root_dir + '/' + args.modelName + '_dev.txt'
                # fwrite = open(fp_write, 'w')
                # for pred_batch_result in pred_batch_results:
                #     for pred_result in pred_batch_result:
                #         # print("pred_result: ", pred_result)
                #         fwrite.write(pred_result + '\n')
                args.best_dev_f1 = f1

                # begin{save the predict prob}
                all_predicts = [list(x['predicts']) for x in outputs]
                all_span_words = [list(x['all_span_word']) for x in outputs]

                # begin{get the label2idx dictionary}
                label2idx = {}
                label2idx_list = args.label2idx_list
                for labidx in label2idx_list:
                    lab, idx = labidx
                    label2idx[lab] = int(idx)
                    # end{get the label2idx dictionary}

                file_prob1 = args.default_root_dir + '/' + args.modelName + '_prob_dev.pkl'
                print("the file path of probs: ", file_prob1)
                fwrite_prob = open(file_prob1, 'wb')
                pickle.dump([label2idx, all_predicts,
                            all_span_words], fwrite_prob)
                # save pytorch model
                file_model1 = args.default_root_dir + '/' + args.modelName + 'best_model_dev.pth'
                print("the file path of model: ", file_model1)
                torch.save(model.state_dict(), file_model1)

                # end{save the predict prob...}

            print("Validation:",  {
                  'val_loss': avg_loss, 'log': tensorboard_logs})

    # TEST the shit out of it
    model.eval()
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
            if args.use_prune:
                span_f1s, pred_label_idx = span_f1_prune(
                    all_span_idxs, predicts, span_label_ltoken, real_span_mask_ltoken)
                # print('pred_label_idx_new: ',pred_label_idx_new.shape)
                # print('predicts: ', predicts.shape)
                # print('pred_label_idx_new: ',pred_label_idx_new)
                # print('predicts: ', predicts)

                batch_preds = get_predict_prune(args, all_span_word, words, pred_label_idx, span_label_ltoken,
                                                all_span_idxs)
            else:
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
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # tensorboard_logs = {'val_loss': avg_loss}
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
        tensorboard_logs[f"span_precision"] = precision
        tensorboard_logs[f"span_recall"] = recall
        tensorboard_logs[f"span_f1"] = f1

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

        print("Test: ", {'val_loss': avg_loss, 'log': tensorboard_logs})

    # LIGHTNING CODE ----
    # trainer = Trainer.from_argparse_args(
    #     args,
    #     # checkpoint_callback=checkpoint_callback
    # )


    # trainer.fit(model)
    # trainer.test() # TO BE CHANGED
if __name__ == '__main__':
    # run_dataloader()
    main()


# n = torch.bincount(label)
# n_i = n[label] - 1

# sameEmbedding = label.unsqueeze(0) == label.unsqueeze(1)
# sameEmbedding.fill_diagonal_(0)
# sameEmbedding = sameEmbedding.nonzero()

# if sameEmbedding.size(0) > 0:
#     inner_sum = torch.log(cosine_sim[sameEmbedding[:, 0], sameEmbedding[:, 1]] / row_sum[sameEmbedding[:, 0]])
#     contrastive_loss += inner_sum.sum() / (-n_i.float())
