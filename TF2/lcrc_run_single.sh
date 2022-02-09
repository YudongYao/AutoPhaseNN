#!/bin/bash
#SBATCH --job-name=phase
#SBATCH --account=AutoPhase
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

module load gcc/9.2.0-r4tyw54
module load cuda/11.0.2-4szlv2t
module load cudnn/8.1.1.33
module load anaconda3/2020.11
source activate ../../tensorflow


SCRIPT='./train_network_unsup_3D.py'
output='logging.txt'

# set parameters
batch_size=16
n_epoch=100
loss_type='mae'
lr_type='Step'
Initlr=1e-3
T=0.1
train_size=50000
result_path=./results/T${T}_${loss_type}_batch${batch_size}_${lr_type}_Init${Initlr}/
data_path=./CDI_simulation_upsamp_noise/
note="train test on augmentation dataset 50k"
n_workers=48
gpu_num=8
note="${note// /_}"


mkdir -p $result_path
python $SCRIPT --notes $note --loss_type $loss_type --lr_type $lr_type --Initlr $Initlr --T $T --train_size $train_size --OutputFolder $result_path --DataFolder $data_path --num_workers $n_workers --batch_size $batch_size --epoch $n_epoch --gpu_num $gpu_num|& tee -a $result_path$output