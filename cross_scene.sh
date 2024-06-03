#!/usr/bin/env bash
GPU_ID=0
data_dir=/root/autodl-fs/datasets/CrossDomain/CrossScene/
output_dir=./logs_CrossScene
mkdir  $output_dir

#algorithm=('ERM' 'DANN' 'BNM' 'DAAN' 'DAN' 'DeepCoral' 'DSAN')
algorithm=('FCAN')
seed=0

for alg in "${algorithm[@]}"
do 
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain AID --tgt_domain CLRS | tee $output_dir/${alg}_A2C.log
CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain AID --tgt_domain CLRS | tee $output_dir/${alg}_A2C_analysis.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain AID --tgt_domain MLRSN | tee $output_dir/${alg}_A2M.log
CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain AID --tgt_domain MLRSN | tee $output_dir/${alg}_A2M_analysis.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain AID --tgt_domain RSSCN7 | tee $output_dir/${alg}_A2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain AID --tgt_domain RSSCN7 | tee $output_dir/${alg}_A2R_analysis.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain CLRS --tgt_domain AID | tee $output_dir/${alg}_C2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain CLRS --tgt_domain AID | tee $output_dir/${alg}_C2A_analysis.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain CLRS --tgt_domain MLRSN | tee $output_dir/${alg}_C2M.log
CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain CLRS --tgt_domain MLRSN | tee $output_dir/${alg}_C2M_analysis.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain CLRS --tgt_domain RSSCN7 | tee $output_dir/${alg}_C2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain CLRS --tgt_domain RSSCN7 | tee $output_dir/${alg}_C2R_analysis.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain MLRSN --tgt_domain AID | tee $output_dir/${alg}_M2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain MLRSN --tgt_domain AID | tee $output_dir/${alg}_M2A_analysis.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain MLRSN --tgt_domain CLRS | tee $output_dir/${alg}_M2C.log
CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain MLRSN --tgt_domain CLRS | tee $output_dir/${alg}_M2C_analysis.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain MLRSN --tgt_domain RSSCN7 | tee $output_dir/${alg}_M2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain MLRSN --tgt_domain RSSCN7 | tee $output_dir/${alg}_M2R_analysis.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain RSSCN7 --tgt_domain AID | tee $output_dir/${alg}_R2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain RSSCN7 --tgt_domain AID | tee $output_dir/${alg}_R2A_analysis.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain RSSCN7 --tgt_domain CLRS | tee $output_dir/${alg}_R2C.log
CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain RSSCN7 --tgt_domain CLRS | tee $output_dir/${alg}_R2C_analysis.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain RSSCN7 --tgt_domain MLRSN | tee $output_dir/${alg}_R2M.log
CUDA_VISIBLE_DEVICES=$GPU_ID python infer.py --config $alg/${alg}.yaml --seed $seed --data_dir $data_dir --src_domain RSSCN7 --tgt_domain MLRSN | tee $output_dir/${alg}_R2M_analysis.log

done