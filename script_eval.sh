#!/bin/bash -l
#SBATCH --output=eval.out
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --constrain='geforce_rtx_2080_ti|geforce_gtx_1080_ti|titan_xp|geforce_gtx_titan_x'

source /itet-stor/shecai/net_scratch/conda/etc/profile.d/conda.sh
conda activate pi-gan
python eval/eval_pix2nerf_opti.py -n srn_chairs \
                              -c conf/exp/srn.conf \
                              -F srn \
                              -D /scratch_net/biwidl212/shecai/datasets/srn_chairs/chairs \
                              -O /home/shecai/Desktop/pixel-nerf/eval_output \
                              --write_compare