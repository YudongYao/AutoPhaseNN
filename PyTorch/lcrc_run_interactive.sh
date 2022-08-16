#!/bin/bash

export PROC_PER_NODE=2
export MASTER_PORT_JOB="28892"
export MASTER_ADDR='gpu2'

source activate pytorch1
echo "Which python?"
which python

echo "Current Directory: "
pwd

SCRIPT='lcrc_train_multiGPU_DDP_fp16.py'

data_path=/lcrc/project/AutoPhase/CDI_simulation_upsamp_aug_220429/
DataSummary='3D_upsamp.txt'

output='logging.txt'

echo "Dataset path $data_path"

# N_NODES=1
# RANKS_PER_NODE=8
# let N_RANKS=${RANKS_PER_NODE}*${N_NODES}

note="supervised on defect free data"
note="${note// /_}"
echo $note

argument_func (){
    local fp16=$1
    local unsupervise=$2
    local use_down_stride=$3
    local use_up_stride=$4
    
    if $fp16; then
        argument="$argument--fp16 "
    fi
    if $unsupervise; then
        argument="$argument--unsupervise "
    fi
    if $use_down_stride; then
        argument="$argument--use_down_stride "
    fi
    if $use_up_stride; then
        argument="$argument--use_up_stride "
    fi
}

fp16=false
unsupervise=false
use_down_stride=true
use_up_stride=false

train_size='50000'
train_perc=0.9
batch_size=96
n_epoch=100
Initlr=1e-4
T=0.1
scale_I=1
loss_type='mae'
lr_type='clr' #'clr', 'step', 'plateau'
optim_type='adam' #'adam', 'adamw'

save_model=10
n_workers=128

result_path=/lcrc/project/AutoPhase/pytorch_test/Unsup${unsupervise}_D${use_down_stride}_U${use_up_stride}_T${T}_${loss_type}_batch${batch_size}_${lr_type}_Init${Initlr}_${optim_type}_scale${scale_I}/

echo "Saving path $result_path"
mkdir -p $result_path


argument=''
argument_func $fp16 $unsupervise  $use_down_stride $use_up_stride

echo "Which python?"
which python
# echo $note
python -m torch.distributed.launch \
    --nproc_per_node=$PROC_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT_JOB} \
    $SCRIPT --notes "$note" \
    --save_model $save_model --device cuda --OutputFolder $result_path --DataFolder $data_path \
    --num_workers $n_workers --batch_size ${batch_size} --epoch ${n_epoch} --optim_type $optim_type\
    --Initlr $Initlr $argument --train_size $train_size --train_perc $train_perc --lr_type $lr_type\
    --T $T --DataSummary $DataSummary --scale_I $scale_I |& tee -a $result_path$output

