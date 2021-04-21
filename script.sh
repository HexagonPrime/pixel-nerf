#!/bin/bash -l
#SBATCH --output=log/%j.out
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --constrain='titan_xp|geforce_rtx_2080_ti|geforce gtx_1080_ti|titan_xp_collectors_edition'

source /scratch_net/biwidl306/shecai/conda/etc/profile.d/conda.sh
conda activate pixelnerf
#module load libs/cuda
python train/train.py -n dtu_exp -c conf/exp/dtu.conf -D /scratch_net/biwidl306/shecai/rs_dtu_4 -V 3 --gpu_id=0