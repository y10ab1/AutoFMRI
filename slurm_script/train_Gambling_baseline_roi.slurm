#!/bin/bash
#SBATCH --job-name=Gambling_baseline_roi    # create a short name for your job
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



CUDA_VISIBLE_DEVICES=0 python train_Gambling_baseline_roi.py --stage1_model cnn --result_dir result-Gambling/baseline-roi-cnn/subj1 --subject 1 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj1/mask_p0.001.nii.gz & 
CUDA_VISIBLE_DEVICES=1 python train_Gambling_baseline_roi.py --stage1_model cnn --result_dir result-Gambling/baseline-roi-cnn/subj2 --subject 2 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj2/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=2 python train_Gambling_baseline_roi.py --stage1_model cnn --result_dir result-Gambling/baseline-roi-cnn/subj3 --subject 3 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj3/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=3 python train_Gambling_baseline_roi.py --stage1_model cnn --result_dir result-Gambling/baseline-roi-cnn/subj4 --subject 4 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj4/mask_p0.001.nii.gz &
CUDA_VISIBLE_DEVICES=4 python train_Gambling_baseline_roi.py --stage1_model cnn --result_dir result-Gambling/baseline-roi-cnn/subj5 --subject 5 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj5/mask_p0.001.nii.gz &

wait

CUDA_VISIBLE_DEVICES=0 python train_Gambling_baseline_roi.py --stage1_model cnn --result_dir result-Gambling/baseline-roi-cnn/subj6 --subject 6 --n_jobs 8 --mask_file  data/HCP_GAMBLING/subj6/mask_p0.001.nii.gz &  
CUDA_VISIBLE_DEVICES=1 python train_Gambling_baseline_roi.py --stage1_model cnn --result_dir result-Gambling/baseline-roi-cnn/subj7 --subject 7 --n_jobs 8 --mask_file  data/HCP_GAMBLING/subj7/mask_p0.001.nii.gz &  
CUDA_VISIBLE_DEVICES=2 python train_Gambling_baseline_roi.py --stage1_model cnn --result_dir result-Gambling/baseline-roi-cnn/subj8 --subject 8 --n_jobs 8 --mask_file  data/HCP_GAMBLING/subj8/mask_p0.001.nii.gz &  
CUDA_VISIBLE_DEVICES=3 python train_Gambling_baseline_roi.py --stage1_model cnn --result_dir result-Gambling/baseline-roi-cnn/subj9 --subject 9 --n_jobs 8 --mask_file  data/HCP_GAMBLING/subj9/mask_p0.001.nii.gz &  
CUDA_VISIBLE_DEVICES=4 python train_Gambling_baseline_roi.py --stage1_model cnn --result_dir result-Gambling/baseline-roi-cnn/subj10 --subject 10 --n_jobs 8 --mask_file data/HCP_GAMBLING/subj10/mask_p0.001.nii.gz &  

wait

CUDA_VISIBLE_DEVICES=0 python train_Gambling_baseline_roi.py --stage1_model cnn --result_dir result-Gambling/baseline-roi-cnn/subj11 --subject 11 --n_jobs 8 --mask_file  data/HCP_GAMBLING/subj11/mask_p0.001.nii.gz &  
CUDA_VISIBLE_DEVICES=1 python train_Gambling_baseline_roi.py --stage1_model cnn --result_dir result-Gambling/baseline-roi-cnn/subj12 --subject 12 --n_jobs 8 --mask_file  data/HCP_GAMBLING/subj12/mask_p0.001.nii.gz &  
CUDA_VISIBLE_DEVICES=2 python train_Gambling_baseline_roi.py --stage1_model cnn --result_dir result-Gambling/baseline-roi-cnn/subj13 --subject 13 --n_jobs 8 --mask_file  data/HCP_GAMBLING/subj13/mask_p0.001.nii.gz &  

wait

