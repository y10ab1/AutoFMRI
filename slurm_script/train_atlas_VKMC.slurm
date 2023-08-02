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


# CUDA_VISIBLE_DEVICES=0 python train_atlas_VKMC.py --data_dir data/2021_Reconstruction_dataset --subject sub-001 --stage1_model rf --stage2_model rf --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-VKMC-rf-rf-yeo-RandomSearch/sub-001 &
# # CUDA_VISIBLE_DEVICES=1 python train_atlas_VKMC.py --data_dir data/2021_Reconstruction_dataset --subject sub-002 --stage1_model rf --stage2_model rf --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-VKMC-rf-rf-yeo-RandomSearch/sub-002 &
# # CUDA_VISIBLE_DEVICES=2 python train_atlas_VKMC.py --data_dir data/2021_Reconstruction_dataset --subject sub-003 --stage1_model rf --stage2_model rf --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-VKMC-rf-rf-yeo-RandomSearch/sub-003 &
# CUDA_VISIBLE_DEVICES=3 python train_atlas_VKMC.py --data_dir data/2021_Reconstruction_dataset --subject sub-005 --stage1_model rf --stage2_model rf --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-VKMC-rf-rf-yeo-RandomSearch/sub-005 &
# # CUDA_VISIBLE_DEVICES=4 python train_atlas_VKMC.py --data_dir data/2021_Reconstruction_dataset --subject sub-006 --stage1_model rf --stage2_model rf --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-VKMC-rf-rf-yeo-RandomSearch/sub-006 &
# CUDA_VISIBLE_DEVICES=4 python train_atlas_VKMC.py --data_dir data/2021_Reconstruction_dataset --subject sub-105 --stage1_model rf --stage2_model rf --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-VKMC-rf-rf-yeo-RandomSearch/sub-105 &

CUDA_VISIBLE_DEVICES=3 python train_atlas_VKMC.py --data_dir data/2021_Reconstruction_dataset --subject sub-105 --stage1_model xgb --stage2_model xgb --max_patches_ratio 0.5 --n_jobs 8 --atlas_name yeo --result_dir result-VKMC-xgb-xgb-yeo-RandomSearch-volume/sub-105 &

wait

# python train_atlas_VKMC_.py --data_dir data/2021_Reconstruction_dataset --subject sub-001 \
#                                                                                     sub-002 \
#                                                                                     sub-003 \
#                                                                                     sub-005 \
#                                                                                     sub-006 \
#                                                                                     sub-007 \
#                                                                                     sub-101 \
#                                                                                     sub-102 \
#                                                                                     sub-103 \
#                                                                                     sub-104 \
#                                                                                     sub-105 \
#                                                                                     sub-106 \
#                                                                                     sub-107 \
#                                                                                     sub-201 \
#                                                                                     sub-202 \
#                                                                                     sub-203 \
#                                                                                     sub-204 \
#                                                                                     sub-205 \
#                                                                                     sub-207 \
#                                                                                     sub-208 --stage1_model xgb --stage2_model xgb --max_patches_ratio 0.5 --n_jobs 8 --atlas_name yeo --result_dir result-VKMC-xgb-xgb-yeo-RandomSearch/sub-multiple &
# wait


