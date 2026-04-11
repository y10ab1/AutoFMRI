#!/bin/bash
#SBATCH --job-name=Gambling_baseline_roi_anova    # create a short name for your job
#SBATCH --account=MST111218                     # project id
#SBATCH --nodes=1 --ntasks-per-node=4           # node count
#SBATCH --gres=gpu:5                            # number of gpus per node
#SBATCH --time=24:00:00                         # total run time limit (HH:MM:SS)
#SBATCH --mail-user your-email@example.com
#SBATCH --mail-type=begin                       # send mail when job begins
#SBATCH --mail-type=end                         # send mail when job ends
#SBATCH --mail-type=fail                        # send mail if job fails

lscpu
nvidia-smi

ml r conda-LSA-TW2
ml cuda/11.7

conda deactivate
conda activate autofmri



CUDA_VISIBLE_DEVICES=0 python train_Gambling_baseline_roi_anova.py --stage1_model rf --subject 1 --result_dir result-Gambling/baseline-roi-anova-rf/subj1 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj1/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=1 python train_Gambling_baseline_roi_anova.py --stage1_model rf --subject 2 --result_dir result-Gambling/baseline-roi-anova-rf/subj2 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj2/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=2 python train_Gambling_baseline_roi_anova.py --stage1_model rf --subject 3 --result_dir result-Gambling/baseline-roi-anova-rf/subj3 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj3/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=3 python train_Gambling_baseline_roi_anova.py --stage1_model rf --subject 4 --result_dir result-Gambling/baseline-roi-anova-rf/subj4 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj4/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=4 python train_Gambling_baseline_roi_anova.py --stage1_model rf --subject 5 --result_dir result-Gambling/baseline-roi-anova-rf/subj5 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj5/mask_p0.001.nii.gz &

wait

CUDA_VISIBLE_DEVICES=0 python train_Gambling_baseline_roi_anova.py --stage1_model rf --subject 6 --result_dir result-Gambling/baseline-roi-anova-rf/subj6 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj6/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=1 python train_Gambling_baseline_roi_anova.py --stage1_model rf --subject 7 --result_dir result-Gambling/baseline-roi-anova-rf/subj7 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj7/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=2 python train_Gambling_baseline_roi_anova.py --stage1_model rf --subject 8 --result_dir result-Gambling/baseline-roi-anova-rf/subj8 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj8/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=3 python train_Gambling_baseline_roi_anova.py --stage1_model rf --subject 9 --result_dir result-Gambling/baseline-roi-anova-rf/subj9 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj9/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=4 python train_Gambling_baseline_roi_anova.py --stage1_model rf --subject 10 --result_dir result-Gambling/baseline-roi-anova-rf/subj10 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj10/mask_p0.001.nii.gz &

wait

CUDA_VISIBLE_DEVICES=0 python train_Gambling_baseline_roi_anova.py --stage1_model rf --subject 11 --result_dir result-Gambling/baseline-roi-anova-rf/subj11 --n_jobs 2 --mask_file data/HCP_GAMBLING/subj11/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=1 python train_Gambling_baseline_roi_anova.py --stage1_model rf --subject 12 --result_dir result-Gambling/baseline-roi-anova-rf/subj12 --n_jobs 2 --mask_file data/HCP_GAMBLING/subj12/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=2 python train_Gambling_baseline_roi_anova.py --stage1_model rf --subject 13 --result_dir result-Gambling/baseline-roi-anova-rf/subj13 --n_jobs 2 --mask_file data/HCP_GAMBLING/subj13/mask_p0.001.nii.gz &

wait