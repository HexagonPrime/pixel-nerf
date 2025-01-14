#!/bin/bash -l
#SBATCH --output=log/dtu.out
#SBATCH --mem=50G
#SBATCH --gres=gpu:6
#SBATCH --constrain='geforce_rtx_2080_ti|geforce_gtx_1080_ti|titan_xp|geforce_gtx_titan_x'

source /scratch_net/biwidl306/shecai/conda/etc/profile.d/conda.sh
conda activate pixelnerf
#module load libs/cuda
python train/train.py -n dtu_exp -c conf/exp/dtu.conf -D /scratch_net/biwidl306/shecai/rs_dtu_4 -V 3 --gpu_id='0 1 2 3 4 5' -R 20000 -B 4