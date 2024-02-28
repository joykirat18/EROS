export PYTHONPATH="$PWD"

DATA_DIR="data/fepd_filtered"
PRETRAINED="bert-large-uncased"
BERT_DIR=""${PRETRAINED}




dataname=fepd
n_class=10
BERT_DROPOUT=0.2
MODEL_DROPOUT=0.2
LR=1e-5
MAXLEN=512
MAXNORM=1.0
batchSize=4
max_spanLen=10
tokenLen_emb_dim=200
spanLen_emb_dim=300
morph_emb_dim=300


use_prune=False
use_spanLen=True
use_morph=False
use_span_weight=False
neg_span_weight=0.5
gpus="0,"
device="cuda:0"



max_epochs=30
modelName="spanner_"${PRETRAINED}_spMLen${max_span_len}_usePrune${use_prune}_useSpLen${use_spanLen}_useSpMorph${use_morph}_SpWt${use_span_weight}_value${neg_span_weight}
idtest=${dataname}_${modelName}
param_name=epoch${max_epochs}_batchsize${batchSize}_lr${LR}_maxlen${MAXLEN}

OUTPUT_DIR="train_logs/$dataname/${modelName}"
mkdir -p $OUTPUT_DIR

python3 trainer.py \
--dataname $dataname \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--bert_max_length $MAXLEN \
--batch_size $batchSize \
--device $device \
--gpus=$gpus \
--precision=16 \
--lr $LR \
--val_check_interval 0.25 \
--accumulate_grad_batches 1 \
--default_root_dir $OUTPUT_DIR \
--model_dropout $MODEL_DROPOUT \
--bert_dropout $BERT_DROPOUT \
--max_epochs $max_epochs \
--n_class $n_class \
--max_spanLen $max_spanLen \
--tokenLen_emb_dim $tokenLen_emb_dim \
--modelName $modelName \
--spanLen_emb_dim $spanLen_emb_dim \
--morph_emb_dim $morph_emb_dim \
--use_prune $use_prune \
--use_spanLen $use_spanLen \
--use_morph $use_morph \
--use_span_weight $use_span_weight \
--neg_span_weight $neg_span_weight \
--param_name $param_name \
--gradient_clip_val $MAXNORM \
--optimizer "adamw" > epoch_results_Contrastive.txt



