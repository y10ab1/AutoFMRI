#!/bin/bash
#SBATCH --job-name=Language_baseline_roi_rfe    # create a short name for your job
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



CUDA_VISIBLE_DEVICES=0 python train_Language_baseline_roi_rfe.py --stage2_model cnn --subject 1 --result_dir result-Language/baseline-roi-rfe-cnn/subj1 --n_jobs 8 --mask_file data/HCP_LANGUAGE/subj1/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=1 python train_Language_baseline_roi_rfe.py --stage2_model cnn --subject 2 --result_dir result-Language/baseline-roi-rfe-cnn/subj2 --n_jobs 8 --mask_file data/HCP_LANGUAGE/subj2/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=2 python train_Language_baseline_roi_rfe.py --stage2_model cnn --subject 3 --result_dir result-Language/baseline-roi-rfe-cnn/subj3 --n_jobs 8 --mask_file data/HCP_LANGUAGE/subj3/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=3 python train_Language_baseline_roi_rfe.py --stage2_model cnn --subject 4 --result_dir result-Language/baseline-roi-rfe-cnn/subj4 --n_jobs 8 --mask_file data/HCP_LANGUAGE/subj4/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=4 python train_Language_baseline_roi_rfe.py --stage2_model cnn --subject 5 --result_dir result-Language/baseline-roi-rfe-cnn/subj5 --n_jobs 8 --mask_file data/HCP_LANGUAGE/subj5/mask_p0.001.nii.gz &

wait

CUDA_VISIBLE_DEVICES=0 python train_Language_baseline_roi_rfe.py --stage2_model cnn --subject 6 --result_dir result-Language/baseline-roi-rfe-cnn/subj6 --n_jobs 8 --mask_file data/HCP_LANGUAGE/subj6/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=1 python train_Language_baseline_roi_rfe.py --stage2_model cnn --subject 7 --result_dir result-Language/baseline-roi-rfe-cnn/subj7 --n_jobs 8 --mask_file data/HCP_LANGUAGE/subj7/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=2 python train_Language_baseline_roi_rfe.py --stage2_model cnn --subject 8 --result_dir result-Language/baseline-roi-rfe-cnn/subj8 --n_jobs 8 --mask_file data/HCP_LANGUAGE/subj8/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=3 python train_Language_baseline_roi_rfe.py --stage2_model cnn --subject 9 --result_dir result-Language/baseline-roi-rfe-cnn/subj9 --n_jobs 8 --mask_file data/HCP_LANGUAGE/subj9/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=4 python train_Language_baseline_roi_rfe.py --stage2_model cnn --subject 10 --result_dir result-Language/baseline-roi-rfe-cnn/subj10 --n_jobs 8 --mask_file data/HCP_LANGUAGE/subj10/mask_p0.001.nii.gz &

wait

CUDA_VISIBLE_DEVICES=0 python train_Language_baseline_roi_rfe.py --stage2_model cnn --subject 11 --result_dir result-Language/baseline-roi-rfe-cnn/subj11 --n_jobs 2 --mask_file data/HCP_LANGUAGE/subj11/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=1 python train_Language_baseline_roi_rfe.py --stage2_model cnn --subject 12 --result_dir result-Language/baseline-roi-rfe-cnn/subj12 --n_jobs 2 --mask_file data/HCP_LANGUAGE/subj12/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=2 python train_Language_baseline_roi_rfe.py --stage2_model cnn --subject 13 --result_dir result-Language/baseline-roi-rfe-cnn/subj13 --n_jobs 2 --mask_file data/HCP_LANGUAGE/subj13/mask_p0.001.nii.gz &

wait