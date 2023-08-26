#!/bin/bash
#SBATCH --job-name=VKMC_baseline_roi            # create a short name for your job
#SBATCH --account=MST111218                     # project id
#SBATCH --nodes=1 --ntasks-per-node=4           # node count
#SBATCH --gres=gpu:5                            # number of gpus per node
#SBATCH --time=24:00:00                         # total run time limit (HH:MM:SS)
#SBATCH --mail-user brian880825@gmail.com
#SBATCH --mail-type=begin                       # send mail when job begins
#SBATCH --mail-type=end                         # send mail when job ends
#SBATCH --mail-type=fail                        # send mail if job fails

lscpu
nvidia-smi

ml r conda-LSA-TW2
ml cuda/11.7

conda deactivate
conda activate autofmri



CUDA_VISIBLE_DEVICES=0 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-001 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-001 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=1 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-002 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-002 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=2 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-003 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-003 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=3 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-005 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-005 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=4 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-006 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-006 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &

wait

CUDA_VISIBLE_DEVICES=0 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-007 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-007 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=1 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-101 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-101 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=2 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-102 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-102 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=3 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-103 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-103 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=4 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-104 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-104 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &

wait

CUDA_VISIBLE_DEVICES=0 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-105 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-105 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=1 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-106 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-106 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=2 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-107 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-107 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=3 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-201 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-201 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=4 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-202 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-202 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &

wait

CUDA_VISIBLE_DEVICES=0 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-203 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-203 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=1 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-204 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-204 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=2 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-205 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-205 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=3 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-207 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-207 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &
CUDA_VISIBLE_DEVICES=4 python train_VKMC_baseline_roi.py --stage1_model cnn --num_classes 5 --data_dir data/2021_Reconstruction_dataset --subject sub-208 --result_dir result-VKMC/baseline-roi-atlas-yeo-cnn/sub-208 --mask_file data/2021_Reconstruction_dataset/masks/ACmask_yeo.nii &

wait
