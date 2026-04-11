#!/bin/bash
#SBATCH --job-name=Emo_baseline_roi    # create a short name for your job
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



CUDA_VISIBLE_DEVICES=0 python train_Emo_baseline_roi.py --stage1_model rf --result_dir result-Emo/baseline-roi-rf/subj1 --subject 1 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=1 python train_Emo_baseline_roi.py --stage1_model rf --result_dir result-Emo/baseline-roi-rf/subj2 --subject 2 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=2 python train_Emo_baseline_roi.py --stage1_model rf --result_dir result-Emo/baseline-roi-rf/subj3 --subject 3 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=3 python train_Emo_baseline_roi.py --stage1_model rf --result_dir result-Emo/baseline-roi-rf/subj4 --subject 4 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=4 python train_Emo_baseline_roi.py --stage1_model rf --result_dir result-Emo/baseline-roi-rf/subj5 --subject 5 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &

wait

CUDA_VISIBLE_DEVICES=0 python train_Emo_baseline_roi.py --stage1_model svm --result_dir result-Emo/baseline-roi-svm/subj1 --subject 1 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=1 python train_Emo_baseline_roi.py --stage1_model svm --result_dir result-Emo/baseline-roi-svm/subj2 --subject 2 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=2 python train_Emo_baseline_roi.py --stage1_model svm --result_dir result-Emo/baseline-roi-svm/subj3 --subject 3 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=3 python train_Emo_baseline_roi.py --stage1_model svm --result_dir result-Emo/baseline-roi-svm/subj4 --subject 4 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=4 python train_Emo_baseline_roi.py --stage1_model svm --result_dir result-Emo/baseline-roi-svm/subj5 --subject 5 --n_jobs 8 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &

wait

CUDA_VISIBLE_DEVICES=0 python train_Emo_baseline_roi.py --stage1_model cnn --result_dir result-Emo/baseline-roi-cnn/subj1 --subject 1 --n_jobs 2 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=1 python train_Emo_baseline_roi.py --stage1_model cnn --result_dir result-Emo/baseline-roi-cnn/subj2 --subject 2 --n_jobs 2 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=2 python train_Emo_baseline_roi.py --stage1_model cnn --result_dir result-Emo/baseline-roi-cnn/subj3 --subject 3 --n_jobs 2 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=3 python train_Emo_baseline_roi.py --stage1_model cnn --result_dir result-Emo/baseline-roi-cnn/subj4 --subject 4 --n_jobs 2 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &
CUDA_VISIBLE_DEVICES=4 python train_Emo_baseline_roi.py --stage1_model cnn --result_dir result-Emo/baseline-roi-cnn/subj5 --subject 5 --n_jobs 2 --mask_file data/fmri-dataset-for-emotion-recognition/masks/mean_tfce_corrp_tstat1.nii &

wait