#!/bin/bash
#SBATCH --job-name=Emo_baseline_roi_anova    # create a short name for your job
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



CUDA_VISIBLE_DEVICES=0 python train_Emo_baseline_roi_anova.py --stage1_model rf --subject 1 --result_dir result-Emo/baseline-roi-anova-rf/subj1 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=1 python train_Emo_baseline_roi_anova.py --stage1_model rf --subject 2 --result_dir result-Emo/baseline-roi-anova-rf/subj2 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=2 python train_Emo_baseline_roi_anova.py --stage1_model rf --subject 3 --result_dir result-Emo/baseline-roi-anova-rf/subj3 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=3 python train_Emo_baseline_roi_anova.py --stage1_model rf --subject 4 --result_dir result-Emo/baseline-roi-anova-rf/subj4 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=4 python train_Emo_baseline_roi_anova.py --stage1_model rf --subject 5 --result_dir result-Emo/baseline-roi-anova-rf/subj5 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &

wait

CUDA_VISIBLE_DEVICES=0 python train_Emo_baseline_roi_anova.py --stage1_model svm --subject 1 --result_dir result-Emo/baseline-roi-anova-svm/subj1 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=1 python train_Emo_baseline_roi_anova.py --stage1_model svm --subject 2 --result_dir result-Emo/baseline-roi-anova-svm/subj2 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=2 python train_Emo_baseline_roi_anova.py --stage1_model svm --subject 3 --result_dir result-Emo/baseline-roi-anova-svm/subj3 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=3 python train_Emo_baseline_roi_anova.py --stage1_model svm --subject 4 --result_dir result-Emo/baseline-roi-anova-svm/subj4 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=4 python train_Emo_baseline_roi_anova.py --stage1_model svm --subject 5 --result_dir result-Emo/baseline-roi-anova-svm/subj5 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &

wait

CUDA_VISIBLE_DEVICES=0 python train_Emo_baseline_roi_anova.py --stage1_model cnn --subject 1 --result_dir result-Emo/baseline-roi-anova-cnn/subj1 --n_jobs 2 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=1 python train_Emo_baseline_roi_anova.py --stage1_model cnn --subject 2 --result_dir result-Emo/baseline-roi-anova-cnn/subj2 --n_jobs 2 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=2 python train_Emo_baseline_roi_anova.py --stage1_model cnn --subject 3 --result_dir result-Emo/baseline-roi-anova-cnn/subj3 --n_jobs 2 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=3 python train_Emo_baseline_roi_anova.py --stage1_model cnn --subject 4 --result_dir result-Emo/baseline-roi-anova-cnn/subj4 --n_jobs 2 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=4 python train_Emo_baseline_roi_anova.py --stage1_model cnn --subject 5 --result_dir result-Emo/baseline-roi-anova-cnn/subj5 --n_jobs 2 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &

wait