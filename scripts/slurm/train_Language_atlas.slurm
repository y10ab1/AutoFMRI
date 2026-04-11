#!/bin/bash
#SBATCH --job-name=Language_atlas    # create a short name for your job
#SBATCH --account=MST111218                     # project id
#SBATCH --nodes=1 --ntasks-per-node=20           # node count
#SBATCH --gres=gpu:5                            # number of gpus per node
#SBATCH --time=24:00:00                         # total run time limit (HH:MM:SS)
#SBATCH --mail-user your-email@example.com
#SBATCH --mail-type=begin                       # send mail when job begins
#SBATCH --mail-type=end                         # send mail when job ends
#SBATCH --mail-type=fail                        # send mail if job fails

lscpu
nvidia-smi

ml purge
ml miniconda3
ml cuda/11.7

conda activate autofmri



CUDA_VISIBLE_DEVICES=0 python train_Language_atlas.py --topk_percent_shap 0.8 --subject 1  --stage2_model cnn --max_patches_ratio 0.5 --n_jobs 4 --atlas_name yeo --result_dir result-Language/autofmri-rf-cnn/subj1 &
CUDA_VISIBLE_DEVICES=1 python train_Language_atlas.py --topk_percent_shap 0.8 --subject 2  --stage2_model cnn --max_patches_ratio 0.5 --n_jobs 4 --atlas_name yeo --result_dir result-Language/autofmri-rf-cnn/subj2 &
CUDA_VISIBLE_DEVICES=2 python train_Language_atlas.py --topk_percent_shap 0.8 --subject 3  --stage2_model cnn --max_patches_ratio 0.5 --n_jobs 4 --atlas_name yeo --result_dir result-Language/autofmri-rf-cnn/subj3 &
CUDA_VISIBLE_DEVICES=3 python train_Language_atlas.py --topk_percent_shap 0.8 --subject 4  --stage2_model cnn --max_patches_ratio 0.5 --n_jobs 4 --atlas_name yeo --result_dir result-Language/autofmri-rf-cnn/subj4 &
CUDA_VISIBLE_DEVICES=4 python train_Language_atlas.py --topk_percent_shap 0.8 --subject 5  --stage2_model cnn --max_patches_ratio 0.5 --n_jobs 4 --atlas_name yeo --result_dir result-Language/autofmri-rf-cnn/subj5 &

wait

CUDA_VISIBLE_DEVICES=0 python train_Language_atlas.py --topk_percent_shap 0.8 --subject 6  --stage2_model cnn --max_patches_ratio 0.5 --n_jobs 4 --atlas_name yeo --result_dir result-Language/autofmri-rf-cnn/subj6 &
CUDA_VISIBLE_DEVICES=1 python train_Language_atlas.py --topk_percent_shap 0.8 --subject 7  --stage2_model cnn --max_patches_ratio 0.5 --n_jobs 4 --atlas_name yeo --result_dir result-Language/autofmri-rf-cnn/subj7 &
CUDA_VISIBLE_DEVICES=2 python train_Language_atlas.py --topk_percent_shap 0.8 --subject 8  --stage2_model cnn --max_patches_ratio 0.5 --n_jobs 4 --atlas_name yeo --result_dir result-Language/autofmri-rf-cnn/subj8 &
CUDA_VISIBLE_DEVICES=3 python train_Language_atlas.py --topk_percent_shap 0.8 --subject 9  --stage2_model cnn --max_patches_ratio 0.5 --n_jobs 4 --atlas_name yeo --result_dir result-Language/autofmri-rf-cnn/subj9 &
CUDA_VISIBLE_DEVICES=4 python train_Language_atlas.py --topk_percent_shap 0.8 --subject 10  --stage2_model cnn --max_patches_ratio 0.5 --n_jobs 4 --atlas_name yeo --result_dir result-Language/autofmri-rf-cnn/subj10 &

wait

CUDA_VISIBLE_DEVICES=0 python train_Language_atlas.py --topk_percent_shap 0.8 --subject 11  --stage2_model cnn --max_patches_ratio 0.5 --n_jobs 4 --atlas_name yeo --result_dir result-Language/autofmri-rf-cnn/subj11 &
CUDA_VISIBLE_DEVICES=1 python train_Language_atlas.py --topk_percent_shap 0.8 --subject 12  --stage2_model cnn --max_patches_ratio 0.5 --n_jobs 4 --atlas_name yeo --result_dir result-Language/autofmri-rf-cnn/subj12 &
CUDA_VISIBLE_DEVICES=2 python train_Language_atlas.py --topk_percent_shap 0.8 --subject 13  --stage2_model cnn --max_patches_ratio 0.5 --n_jobs 4 --atlas_name yeo --result_dir result-Language/autofmri-rf-cnn/subj13 &

wait

