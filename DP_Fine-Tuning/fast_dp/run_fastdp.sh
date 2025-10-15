#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=64g                     # allocate 32 GB of memory
#SBATCH -J "deep_speed_run_fastDP"              # name of the job
#SBATCH -o deep_speed_fastdp_run_%j.out         # name of the output file
#SBATCH -e deep_speed_fastdp_run_%j.err         # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 10:00:00                   # time limit of 12 hours
#SBATCH --gres=gpu:H100:2             # request 1 H200 GPU

cd $SLURM_SUBMIT_DIR/..

module load python/3.10.2/mqmlxcf
module load cuda/12.4.0/3mdaov5

python -m venv env
source env/bin/activate

pip install --upgrade pip
pip install git+https://github.com/awslabs/fast-differential-privacy.git
pip install -U "huggingface_hub[cli]"
pip install numpy
pip install torch
pip install transformers
pip install datasets
pip install tqdm
pip install scikit-learn
pip install accelerate
pip install peft
pip install deepspeed

deepspeed --num_gpus=2 fast_dp/fastdp.py --deepspeed_config ds_config.json

