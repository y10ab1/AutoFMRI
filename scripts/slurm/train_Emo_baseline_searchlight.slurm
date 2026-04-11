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



CUDA_VISIBLE_DEVICES=0 python train_Emo_baseline_searchlight.py --subject 1 --stage1_model rf --n_jobs 8 --result_dir result-Emo/baseline-searchlight-rf/sub-01 &
CUDA_VISIBLE_DEVICES=1 python train_Emo_baseline_searchlight.py --subject 2 --stage1_model rf --n_jobs 8 --result_dir result-Emo/baseline-searchlight-rf/sub-02 &
CUDA_VISIBLE_DEVICES=2 python train_Emo_baseline_searchlight.py --subject 3 --stage1_model rf --n_jobs 8 --result_dir result-Emo/baseline-searchlight-rf/sub-03 &
CUDA_VISIBLE_DEVICES=3 python train_Emo_baseline_searchlight.py --subject 4 --stage1_model rf --n_jobs 8 --result_dir result-Emo/baseline-searchlight-rf/sub-04 &
CUDA_VISIBLE_DEVICES=4 python train_Emo_baseline_searchlight.py --subject 5 --stage1_model rf --n_jobs 8 --result_dir result-Emo/baseline-searchlight-rf/sub-05 &

wait

CUDA_VISIBLE_DEVICES=0 python train_Emo_baseline_searchlight.py --subject 1 --stage1_model svm --n_jobs 8 --result_dir result-Emo/baseline-searchlight-svm/sub-01 &
CUDA_VISIBLE_DEVICES=1 python train_Emo_baseline_searchlight.py --subject 2 --stage1_model svm --n_jobs 8 --result_dir result-Emo/baseline-searchlight-svm/sub-02 &
CUDA_VISIBLE_DEVICES=2 python train_Emo_baseline_searchlight.py --subject 3 --stage1_model svm --n_jobs 8 --result_dir result-Emo/baseline-searchlight-svm/sub-03 &
CUDA_VISIBLE_DEVICES=3 python train_Emo_baseline_searchlight.py --subject 4 --stage1_model svm --n_jobs 8 --result_dir result-Emo/baseline-searchlight-svm/sub-04 &
CUDA_VISIBLE_DEVICES=4 python train_Emo_baseline_searchlight.py --subject 5 --stage1_model svm --n_jobs 8 --result_dir result-Emo/baseline-searchlight-svm/sub-05 &

wait

CUDA_VISIBLE_DEVICES=0 python train_Emo_baseline_searchlight.py --subject 1 --stage1_model cnn --n_classes 6 --n_jobs 8 --result_dir result-Emo/baseline-searchlight-cnn/sub-01 &
CUDA_VISIBLE_DEVICES=1 python train_Emo_baseline_searchlight.py --subject 2 --stage1_model cnn --n_classes 6 --n_jobs 8 --result_dir result-Emo/baseline-searchlight-cnn/sub-02 &
CUDA_VISIBLE_DEVICES=2 python train_Emo_baseline_searchlight.py --subject 3 --stage1_model cnn --n_classes 6 --n_jobs 8 --result_dir result-Emo/baseline-searchlight-cnn/sub-03 &
CUDA_VISIBLE_DEVICES=3 python train_Emo_baseline_searchlight.py --subject 4 --stage1_model cnn --n_classes 6 --n_jobs 8 --result_dir result-Emo/baseline-searchlight-cnn/sub-04 &
CUDA_VISIBLE_DEVICES=4 python train_Emo_baseline_searchlight.py --subject 5 --stage1_model cnn --n_classes 6 --n_jobs 8 --result_dir result-Emo/baseline-searchlight-cnn/sub-05 &

wait

