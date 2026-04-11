#!/bin/bash
#SBATCH --job-name=Haxby_atlas    # create a short name for your job
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


CUDA_VISIBLE_DEVICES=0 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj1 --subject 1 --stage1_model rf --stage2_model rf --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-rf-test/subj1 &
CUDA_VISIBLE_DEVICES=1 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj2 --subject 2 --stage1_model rf --stage2_model rf --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-rf-test/subj2 &
CUDA_VISIBLE_DEVICES=2 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj3 --subject 3 --stage1_model rf --stage2_model rf --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-rf-test/subj3 &
CUDA_VISIBLE_DEVICES=3 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj4 --subject 4 --stage1_model rf --stage2_model rf --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-rf-test/subj4 &
CUDA_VISIBLE_DEVICES=4 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj5 --subject 5 --stage1_model rf --stage2_model rf --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-rf-test/subj5 &

wait

CUDA_VISIBLE_DEVICES=0 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj1 --subject 1 --stage1_model rf --stage2_model svm --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-svm-test/subj1 &
CUDA_VISIBLE_DEVICES=1 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj2 --subject 2 --stage1_model rf --stage2_model svm --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-svm-test/subj2 &
CUDA_VISIBLE_DEVICES=2 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj3 --subject 3 --stage1_model rf --stage2_model svm --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-svm-test/subj3 &
CUDA_VISIBLE_DEVICES=3 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj4 --subject 4 --stage1_model rf --stage2_model svm --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-svm-test/subj4 &
CUDA_VISIBLE_DEVICES=4 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj5 --subject 5 --stage1_model rf --stage2_model svm --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-svm-test/subj5 &

wait

CUDA_VISIBLE_DEVICES=0 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj1 --subject 1 --stage1_model rf --stage2_model cnn --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-cnn-test/subj1 &
CUDA_VISIBLE_DEVICES=1 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj2 --subject 2 --stage1_model rf --stage2_model cnn --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-cnn-test/subj2 &
CUDA_VISIBLE_DEVICES=2 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj3 --subject 3 --stage1_model rf --stage2_model cnn --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-cnn-test/subj3 &
CUDA_VISIBLE_DEVICES=3 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj4 --subject 4 --stage1_model rf --stage2_model cnn --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-cnn-test/subj4 &
CUDA_VISIBLE_DEVICES=4 python train_Haxby_atlas.py --n_jobs 4 --data_dir data/haxby2001/subj5 --subject 5 --stage1_model rf --stage2_model cnn --max_patches_ratio 0.5 --atlas_name yeo --result_dir result-Haxby/autofmri-rf-cnn-test/subj5 &

wait