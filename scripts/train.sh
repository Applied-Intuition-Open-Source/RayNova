#!/usr/bin/env bash

set -x

# set dist args
SINGLE=1
nproc_per_node=${ARNOLD_WORKER_GPU}

if [ ! -z "$SINGLE" ] && [ "$SINGLE" != "0" ]; then
  echo "[single node alone] SINGLE=$SINGLE"
  nnodes=1
  node_rank=0
  nproc_per_node=1
  master_addr=127.0.0.1
  master_port=12342
else
  MASTER_NODE_ID=0
  nnodes=${ARNOLD_WORKER_NUM}
  node_rank=${ARNOLD_ID}
  master_addr="METIS_WORKER_${MASTER_NODE_ID}_HOST"
  master_addr=${!master_addr}
  master_port="METIS_WORKER_${MASTER_NODE_ID}_PORT"
  master_port=${!master_port}
  ports=(`echo $master_port | tr ',' ' '`)
  master_port=${ports[0]}
fi

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

BED=checkpoints
LOCAL_OUT=local_output
mkdir -p $BED
mkdir -p $LOCAL_OUT

wandb offline
exp_name=exp_tmp
bed_path=checkpoints/${exp_name}/
data_path='./nuplan_sample/sample_10/'
local_out_path=$LOCAL_OUT/${exp_name}

torchrun \
--nproc_per_node=${nproc_per_node} \
--nnodes=${nnodes} \
--node_rank=${node_rank} \
--master_addr=${master_addr} \
--master_port=${master_port} \
train_raynova.py \
--local_out_path ${local_out_path} \
--bed=${bed_path} \
--data_path=${data_path} \
--exp_name=${exp_name} \
--log_every_iter=True \
--tblr=2e-4 \
--lbs=1 \
--workers=1 \
--save_model_ep_freq=1 \
--auto_resume=1 
