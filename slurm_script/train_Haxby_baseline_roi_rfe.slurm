#!/bin/bash
#SBATCH --job-name=Haxby_baseline_roi_rfe    # create a short name for your job
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



CUDA_VISIBLE_DEVICES=0 python train_Haxby_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj1 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-rf/subj1 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj1_atlas_yeo400.nii &
CUDA_VISIBLE_DEVICES=1 python train_Haxby_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj2 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-rf/subj2 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj2_atlas_yeo400.nii &
CUDA_VISIBLE_DEVICES=2 python train_Haxby_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj3 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-rf/subj3 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj3_atlas_yeo400.nii &
CUDA_VISIBLE_DEVICES=3 python train_Haxby_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj4 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-rf/subj4 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj4_atlas_yeo400.nii &
CUDA_VISIBLE_DEVICES=4 python train_Haxby_baseline_roi_rfe.py --stage1_model rf  --data_dir data/haxby2001/subj5 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-rf/subj5 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj5_atlas_yeo400.nii &

wait

CUDA_VISIBLE_DEVICES=0 python train_Haxby_baseline_roi_rfe.py --stage1_model svm  --data_dir data/haxby2001/subj1 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-svm/subj1 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj1_atlas_yeo400.nii &
CUDA_VISIBLE_DEVICES=1 python train_Haxby_baseline_roi_rfe.py --stage1_model svm  --data_dir data/haxby2001/subj2 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-svm/subj2 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj2_atlas_yeo400.nii &
CUDA_VISIBLE_DEVICES=2 python train_Haxby_baseline_roi_rfe.py --stage1_model svm  --data_dir data/haxby2001/subj3 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-svm/subj3 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj3_atlas_yeo400.nii &
CUDA_VISIBLE_DEVICES=3 python train_Haxby_baseline_roi_rfe.py --stage1_model svm  --data_dir data/haxby2001/subj4 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-svm/subj4 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj4_atlas_yeo400.nii &
CUDA_VISIBLE_DEVICES=4 python train_Haxby_baseline_roi_rfe.py --stage1_model svm  --data_dir data/haxby2001/subj5 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-svm/subj5 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj5_atlas_yeo400.nii &

wait

CUDA_VISIBLE_DEVICES=0 python train_Haxby_baseline_roi_rfe.py --stage1_model cnn  --data_dir data/haxby2001/subj1 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-cnn/subj1 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj1_atlas_yeo400.nii &
CUDA_VISIBLE_DEVICES=1 python train_Haxby_baseline_roi_rfe.py --stage1_model cnn  --data_dir data/haxby2001/subj2 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-cnn/subj2 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj2_atlas_yeo400.nii &
CUDA_VISIBLE_DEVICES=2 python train_Haxby_baseline_roi_rfe.py --stage1_model cnn  --data_dir data/haxby2001/subj3 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-cnn/subj3 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj3_atlas_yeo400.nii &
CUDA_VISIBLE_DEVICES=3 python train_Haxby_baseline_roi_rfe.py --stage1_model cnn  --data_dir data/haxby2001/subj4 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-cnn/subj4 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj4_atlas_yeo400.nii &
CUDA_VISIBLE_DEVICES=4 python train_Haxby_baseline_roi_rfe.py --stage1_model cnn  --data_dir data/haxby2001/subj5 --result_dir result-Haxby/baseline-roi-rfe-atlas-yeo400-cnn/subj5 --mask_file data/haxby2001/wmasks_binary/wmask4_vt_subj5_atlas_yeo400.nii &

wait