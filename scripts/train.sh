#!/bin/bash
########## Instruction ##########
# This script takes three optional environment variables:
# GPU / ADDR / PORT
# e.g. Use gpu 0, 1 and 4 for training, set distributed training
# master address and port to localhost:9901, the command is as follows:
#
# GPU="0,1,4" ADDR=localhost PORT=9901 bash train.sh
#
# Default value: GPU=-1 (use cpu only), ADDR=localhost, PORT=9901
# Note that if your want to run multiple distributed training tasks,
# either the addresses or ports should be different between
# each pair of tasks.
######### end of instruction ##########


########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/..`
echo "Locate the project folder at ${CODE_DIR}"


########## parsing yaml configs ##########
if [ -z $1 ]; then
    echo "Config missing. Usage example: GPU=0,1 bash $0 <config> [optional arguments]"
    exit 1;
fi


########## setup  distributed training ##########
GPU="${GPU:--1}" # default using CPU
MASTER_ADDR="${ADDR:-localhost}"
MASTER_PORT="${PORT:-11451}"
echo "Using GPUs: $GPU"
echo "Master address: ${MASTER_ADDR}, Master port: ${MASTER_PORT}"

export CUDA_VISIBLE_DEVICES=$GPU
export CUDA_LAUNCH_BLOCKING=1
GPU_ARR=(`echo $GPU | tr ',' ' '`)

if [ ${#GPU_ARR[@]} -gt 1 ]; then
    export OMP_NUM_THREADS=2
	PREFIX="torchrun --nproc_per_node=${#GPU_ARR[@]} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --nnodes=1"
else
    PREFIX="python"
fi


########## start training ##########
cd $CODE_DIR
echo "Start training:"
${PREFIX} train.py --gpus "${!GPU_ARR[@]}" --config $@
