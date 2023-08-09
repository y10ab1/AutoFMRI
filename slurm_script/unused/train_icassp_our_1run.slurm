#!/bin/bash
#SBATCH --job-name=AutoFMRI_top1%voxels_top10%cubes_cube25    # create a short name for your job
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



python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj1/first_level_output --subject 1 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj1_rf_rf-1 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj2/first_level_output --subject 2 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj2_rf_rf-1 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj3/first_level_output --subject 3 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj3_rf_rf-1 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj4/first_level_output --subject 4 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj4_rf_rf-1 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj5/first_level_output --subject 5 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj5_rf_rf-1 &
wait

python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj1/first_level_output --subject 1 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj1_rf_rf-2 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj2/first_level_output --subject 2 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj2_rf_rf-2 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj3/first_level_output --subject 3 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj3_rf_rf-2 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj4/first_level_output --subject 4 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj4_rf_rf-2 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj5/first_level_output --subject 5 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj5_rf_rf-2 &
wait

python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj1/first_level_output --subject 1 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj1_rf_rf-3 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj2/first_level_output --subject 2 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj2_rf_rf-3 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj3/first_level_output --subject 3 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj3_rf_rf-3 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj4/first_level_output --subject 4 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj4_rf_rf-3 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj5/first_level_output --subject 5 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj5_rf_rf-3 &
wait

python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj1/first_level_output --subject 1 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj1_rf_rf-4 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj2/first_level_output --subject 2 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj2_rf_rf-4 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj3/first_level_output --subject 3 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj3_rf_rf-4 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj4/first_level_output --subject 4 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj4_rf_rf-4 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj5/first_level_output --subject 5 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj5_rf_rf-4 &
wait

python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj1/first_level_output --subject 1 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj1_rf_rf-5 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj2/first_level_output --subject 2 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj2_rf_rf-5 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj3/first_level_output --subject 3 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj3_rf_rf-5 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj4/first_level_output --subject 4 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj4_rf_rf-5 &
python train.py --cube_size "25 25 25" --cube_stride "10 10 10"  --topk_patches 0.05 --topk_percent_shap 0.01  --stage2_model rf --data_dir data/haxby2001/subj5/first_level_output --subject 5 --result_dir icassp-results-0607-cube25/icassp_haxby2001_subj5_rf_rf-5 &
wait