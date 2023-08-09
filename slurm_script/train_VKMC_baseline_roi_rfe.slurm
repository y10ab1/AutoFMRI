#!/bin/bash
#SBATCH --job-name=AutoFMRI_icassp_5subjects_2-5runs    # create a short name for your job
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



CUDA_VISIBLE_DEVICES=0 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj1 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-001 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=1 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj2 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-002 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=2 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj3 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-003 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=3 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj4 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-005 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=4 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj5 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-006 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &

wait

CUDA_VISIBLE_DEVICES=0 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj1 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-007 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=1 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj2 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-101 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=2 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj3 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-102 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=3 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj4 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-103 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=4 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj5 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-104 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &

wait

CUDA_VISIBLE_DEVICES=0 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj1 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-105 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=1 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj2 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-106 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=2 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj3 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-107 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=3 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj4 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-201 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=4 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj5 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-202 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &

wait

CUDA_VISIBLE_DEVICES=0 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj1 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-203 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=1 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj2 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-204 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=2 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj3 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-205 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=3 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj4 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-207 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &
CUDA_VISIBLE_DEVICES=4 python train_VKMC_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj5 --result_dir result-VKMC/baseline-roi-rfe-vt-rf/sub-208 --mask_file data/2021_Reconstruction_dataset/masks/ACmask.nii &

wait
