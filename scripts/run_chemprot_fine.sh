#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "train.sh <batch_size>"
  exit 1
fi
prefix="chemprotnew"
BATCH_SIZE=$1
#FOLDER=$2
#gpu=$2
#echo "export CUDA_VISIBLE_DEVICES=${gpu}"
#export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

train_datasets="chemprotnew"
test_datasets="chemprotnew"
MODEL_ROOT="checkpoints_chemprotnew"
BERT_PATH="checkpoints_chemprot/mt-dnn-chemprot_adamax_answer_opt1_gc0_ggc1_2020-09-15T0945/strip_model.pt" #####
DATA_DIR="bioRE/mtdnn_weight_new/bert_data/canonical_data/bert_uncased_lower" ####
task_def="bioRE/mtdnn_weight_new/experiments/chemprot_new/chemprot_task_def.yml" #"experiments/euadr/euadr_task_def.yml" 

answer_opt=1
optim="adamax" ##################   
grad_clipping=0
global_grad_clipping=1
lr="5e-5" #5e-5####################### disease 위해 수정함 test -> 1e-5
itw_on=1

model_dir="${MODEL_ROOT}/Chemprot_finetuning"
log_file="${model_dir}/log.log"
 

python train_RE.py --grad_accumulation_step 1 --task_def ${task_def} --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --adv_train --adv_opt 1 --multi_gpu_on
