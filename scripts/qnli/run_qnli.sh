#!/bin/bash
if [[ $# -ne 3 ]]; then
  echo "train.sh <batch_size> <grad_acc_steps> <lr>"
  exit 1
fi
prefix="mt-dnn-qnli"
BATCH_SIZE=$1
GRAD_ACC_STEPS=$2
lr=$3

#echo "export CUDA_VISIBLE_DEVICES=${gpu}"
#export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

train_datasets="qnli"
test_datasets="qnli"
MODEL_ROOT="checkpoints_qnli"
BERT_PATH="checkpoints_qnli/qnli_adamax_5e-5_2020-08-22T1624/qnli_pretrain.pt"
DATA_DIR="data/canonical_data/bert_base_uncased_lower"

answer_opt=0
optim="radam"
grad_clipping=0
global_grad_clipping=1
#lr="5e-5"

model_dir="${MODEL_ROOT}/${lr}_${BATCH_SIZE}_${GRAD_ACC_STEPS}_${prefix}"
log_file="${model_dir}/log.log"
python train.py --grad_accumulation_step $GRAD_ACC_STEPS --task_def experiments/glue/glue_task_def.yml --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --adv_train --adv_opt 1 --multi_gpu_on
