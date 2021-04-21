#!/bin/bash -l
#SBATCH --output=log/%j.out
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --constrain='titan_xp|geforce_gtx_titan_x'

source /scratch/shecai/conda/etc/profile.d/conda.sh
conda activate pixelnerf
#module load libs/cuda
python train/train.py -n dtu_exp -c conf/exp/dtu.conf -D /scratch-second/shecai/rs_dtu_4 -V 3 --gpu_id=0