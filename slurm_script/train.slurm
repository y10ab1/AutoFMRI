#!/bin/bash
#SBATCH --job-name=AutoFMRI    # create a short name for your job
#SBATCH --account=MST111218                     # project id
#SBATCH --nodes=1 --ntasks-per-node=20           # node count
#SBATCH --gres=gpu:5                            # number of gpus per node
#SBATCH --time=24:00:00                         # total run time limit (HH:MM:SS)
#SBATCH --mail-user brian880825@gmail.com
#SBATCH --mail-type=begin                       # send mail when job begins
#SBATCH --mail-type=end                         # send mail when job ends
#SBATCH --mail-type=fail                        # send mail if job fails

lscpu
nvidia-smi

ml purge
ml miniconda3
ml cuda/11.7

conda activate autofmri

#python train.py --cube_size "10 10 10" --topk_patches 40 --data_dir data/haxby2001/subj1/first_level_output --subject 1 --result_dir result-10x10x10-40patches-subj1 &
#python train.py --cube_size "10 10 10" --topk_patches 40 --data_dir data/haxby2001/subj2/first_level_output --subject 2 --result_dir result-10x10x10-40patches-subj2 &
#python train.py --cube_size "10 10 10" --topk_patches 40 --data_dir data/haxby2001/subj3/first_level_output --subject 3 --result_dir result-10x10x10-40patches-subj3 &
#python train.py --cube_size "10 10 10" --topk_patches 40 --data_dir data/haxby2001/subj4/first_level_output --subject 4 --result_dir result-10x10x10-40patches-subj4 &
#python train.py --cube_size "10 10 10" --topk_patches 40 --data_dir data/haxby2001/subj5/first_level_output --subject 5 --result_dir result-10x10x10-40patches-subj5 &

python train.py --cube_size "20 20 20" --topk_patches 5 --data_dir data/haxby2001/subj4/first_level_output --subject 4 --result_dir result-20x20x20-5patches-subj4 &

wait