#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "train.sh <batch_size>"
  exit 1
fi
prefix="mt-dnn-mnli"
BATCH_SIZE=$1
#gpu=$2
#echo "export CUDA_VISIBLE_DEVICES=${gpu}"
#export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

train_datasets="mnli,cola,rte,qqp,qnli,mrpc,sst,stsb"
test_datasets="mnli_matched,mnli_mismatched"
MODEL_ROOT="checkpoints_mnli"
BERT_PATH="mt_dnn_models/bert_model_base_uncased.pt"
DATA_DIR="data/canonical_data/bert_base_uncased_lower"

answer_opt=1
optim="adamax"
epochs=5
lr="5e-5"
grad_clipping=0
global_grad_clipping=1

model_dir="checkpoints_${test_datasets}/${test_datasets}_${optim}_${lr}_${tstr}"
log_file="${model_dir}/log.log"
python train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on --epochs ${epochs} --itw_on True
