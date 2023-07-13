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

CUDA_VISIBLE_DEVICES=0 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj1/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj1_vt_rf-1 --mask_file data/haxby2001/wmasks/wmask4_vt_subj1.nii &
CUDA_VISIBLE_DEVICES=1 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj2/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj2_vt_rf-1 --mask_file data/haxby2001/wmasks/wmask4_vt_subj2.nii &
CUDA_VISIBLE_DEVICES=2 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj3/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj3_vt_rf-1 --mask_file data/haxby2001/wmasks/wmask4_vt_subj3.nii &
CUDA_VISIBLE_DEVICES=3 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj4/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj4_vt_rf-1 --mask_file data/haxby2001/wmasks/wmask4_vt_subj4.nii &
CUDA_VISIBLE_DEVICES=4 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj5/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj5_vt_rf-1 --mask_file data/haxby2001/wmasks/wmask4_vt_subj5.nii &

wait

# CUDA_VISIBLE_DEVICES=0 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj1/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj1_vt_rf-2 --mask_file data/haxby2001/wmasks/wmask4_vt_subj1.nii &
# CUDA_VISIBLE_DEVICES=1 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj2/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj2_vt_rf-2 --mask_file data/haxby2001/wmasks/wmask4_vt_subj2.nii &
# CUDA_VISIBLE_DEVICES=2 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj3/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj3_vt_rf-2 --mask_file data/haxby2001/wmasks/wmask4_vt_subj3.nii &
# CUDA_VISIBLE_DEVICES=3 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj4/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj4_vt_rf-2 --mask_file data/haxby2001/wmasks/wmask4_vt_subj4.nii &
# CUDA_VISIBLE_DEVICES=4 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj5/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj5_vt_rf-2 --mask_file data/haxby2001/wmasks/wmask4_vt_subj5.nii &

# wait

# CUDA_VISIBLE_DEVICES=0 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj1/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj1_vt_rf-3 --mask_file data/haxby2001/wmasks/wmask4_vt_subj1.nii &
# CUDA_VISIBLE_DEVICES=1 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj2/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj2_vt_rf-3 --mask_file data/haxby2001/wmasks/wmask4_vt_subj2.nii &
# CUDA_VISIBLE_DEVICES=2 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj3/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj3_vt_rf-3 --mask_file data/haxby2001/wmasks/wmask4_vt_subj3.nii &
# CUDA_VISIBLE_DEVICES=3 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj4/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj4_vt_rf-3 --mask_file data/haxby2001/wmasks/wmask4_vt_subj4.nii &
# CUDA_VISIBLE_DEVICES=4 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj5/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj5_vt_rf-3 --mask_file data/haxby2001/wmasks/wmask4_vt_subj5.nii &

# wait

# CUDA_VISIBLE_DEVICES=0 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj1/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj1_vt_rf-4 --mask_file data/haxby2001/wmasks/wmask4_vt_subj1.nii &
# CUDA_VISIBLE_DEVICES=1 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj2/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj2_vt_rf-4 --mask_file data/haxby2001/wmasks/wmask4_vt_subj2.nii &
# CUDA_VISIBLE_DEVICES=2 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj3/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj3_vt_rf-4 --mask_file data/haxby2001/wmasks/wmask4_vt_subj3.nii &
# CUDA_VISIBLE_DEVICES=3 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj4/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj4_vt_rf-4 --mask_file data/haxby2001/wmasks/wmask4_vt_subj4.nii &
# CUDA_VISIBLE_DEVICES=4 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj5/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj5_vt_rf-4 --mask_file data/haxby2001/wmasks/wmask4_vt_subj5.nii &

# wait

# CUDA_VISIBLE_DEVICES=0 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj1/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj1_vt_rf-5 --mask_file data/haxby2001/wmasks/wmask4_vt_subj1.nii &
# CUDA_VISIBLE_DEVICES=1 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj2/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj2_vt_rf-5 --mask_file data/haxby2001/wmasks/wmask4_vt_subj2.nii &
# CUDA_VISIBLE_DEVICES=2 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj3/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj3_vt_rf-5 --mask_file data/haxby2001/wmasks/wmask4_vt_subj3.nii &
# CUDA_VISIBLE_DEVICES=3 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj4/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj4_vt_rf-5 --mask_file data/haxby2001/wmasks/wmask4_vt_subj4.nii &
# CUDA_VISIBLE_DEVICES=4 python train_baseline.py --stage1_model rf --mask_threshold 0.05 --data_dir data/haxby2001/subj5/first_level_output --result_dir icassp-results-0605-maskthreshold005/icassp_haxby2001_subj5_vt_rf-5 --mask_file data/haxby2001/wmasks/wmask4_vt_subj5.nii &

# wait

