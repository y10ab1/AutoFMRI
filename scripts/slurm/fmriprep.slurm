#!/bin/bash
#SBATCH --job-name=AutoFMRI    # create a short name for your job
#SBATCH --account=MST111218                     # project id
#SBATCH --nodes=1 --ntasks-per-node=20           # node count
#SBATCH --gres=gpu:5                            # number of gpus per node
#SBATCH --time=24:00:00                         # total run time limit (HH:MM:SS)
#SBATCH --mail-user your-email@example.com
#SBATCH --mail-type=begin                       # send mail when job begins
#SBATCH --mail-type=end                         # send mail when job ends
#SBATCH --mail-type=fail                        # send mail if job fails
cd /work/YOUR_USERNAME
singularity run --cleanenv -B /work/YOUR_USERNAME:/work/YOUR_USERNAME \
                              /work/YOUR_USERNAME/my_images/fmriprep-latest.simg \
                              /work/YOUR_USERNAME/ds001715-download/ \
                              /work/YOUR_USERNAME/ds001715-derivative/ \
                              participant \
                              --fs-no-reconall \
                              --participant-label sub-25 \
                               --fs-license-file /work/YOUR_USERNAME/license.txt &


singularity run --cleanenv -B /work/YOUR_USERNAME:/work/YOUR_USERNAME \
                              /work/YOUR_USERNAME/my_images/fmriprep-latest.simg \
                              /work/YOUR_USERNAME/ds001715-download/ \
                              /work/YOUR_USERNAME/ds001715-derivative/ \
                              participant \
                              --fs-no-reconall \
                              --participant-label sub-26 \
                               --fs-license-file /work/YOUR_USERNAME/license.txt &

singularity run --cleanenv -B /work/YOUR_USERNAME:/work/YOUR_USERNAME \
                              /work/YOUR_USERNAME/my_images/fmriprep-latest.simg \
                              /work/YOUR_USERNAME/ds001715-download/ \
                              /work/YOUR_USERNAME/ds001715-derivative/ \
                              participant \
                              --fs-no-reconall \
                              --participant-label sub-27 \
                               --fs-license-file /work/YOUR_USERNAME/license.txt &


singularity run --cleanenv -B /work/YOUR_USERNAME:/work/YOUR_USERNAME \
                              /work/YOUR_USERNAME/my_images/fmriprep-latest.simg \
                              /work/YOUR_USERNAME/ds001715-download/ \
                              /work/YOUR_USERNAME/ds001715-derivative/ \
                              participant \
                              --fs-no-reconall \
                              --participant-label sub-28 \
                               --fs-license-file /work/YOUR_USERNAME/license.txt &

singularity run --cleanenv -B /work/YOUR_USERNAME:/work/YOUR_USERNAME \
                              /work/YOUR_USERNAME/my_images/fmriprep-latest.simg \
                              /work/YOUR_USERNAME/ds001715-download/ \
                              /work/YOUR_USERNAME/ds001715-derivative/ \
                              participant \
                              --fs-no-reconall \
                              --participant-label sub-29 \
                               --fs-license-file /work/YOUR_USERNAME/license.txt &


singularity run --cleanenv -B /work/YOUR_USERNAME:/work/YOUR_USERNAME \
                              /work/YOUR_USERNAME/my_images/fmriprep-latest.simg \
                              /work/YOUR_USERNAME/ds001715-download/ \
                              /work/YOUR_USERNAME/ds001715-derivative/ \
                              participant \
                              --fs-no-reconall \
                              --participant-label sub-30 \
                               --fs-license-file /work/YOUR_USERNAME/license.txt &


singularity run --cleanenv -B /work/YOUR_USERNAME:/work/YOUR_USERNAME \
                              /work/YOUR_USERNAME/my_images/fmriprep-latest.simg \
                              /work/YOUR_USERNAME/ds001715-download/ \
                              /work/YOUR_USERNAME/ds001715-derivative/ \
                              participant \
                              --fs-no-reconall \
                              --participant-label sub-31 \
                               --fs-license-file /work/YOUR_USERNAME/license.txt &

singularity run --cleanenv -B /work/YOUR_USERNAME:/work/YOUR_USERNAME \
                              /work/YOUR_USERNAME/my_images/fmriprep-latest.simg \
                              /work/YOUR_USERNAME/ds001715-download/ \
                              /work/YOUR_USERNAME/ds001715-derivative/ \
                              participant \
                              --fs-no-reconall \
                              --participant-label sub-32 \
                               --fs-license-file /work/YOUR_USERNAME/license.txt &


singularity run --cleanenv -B /work/YOUR_USERNAME:/work/YOUR_USERNAME \
                              /work/YOUR_USERNAME/my_images/fmriprep-latest.simg \
                              /work/YOUR_USERNAME/ds001715-download/ \
                              /work/YOUR_USERNAME/ds001715-derivative/ \
                              participant \
                              --fs-no-reconall \
                              --participant-label sub-33 \
                               --fs-license-file /work/YOUR_USERNAME/license.txt &

singularity run --cleanenv -B /work/YOUR_USERNAME:/work/YOUR_USERNAME \
                              /work/YOUR_USERNAME/my_images/fmriprep-latest.simg \
                              /work/YOUR_USERNAME/ds001715-download/ \
                              /work/YOUR_USERNAME/ds001715-derivative/ \
                              participant \
                              --fs-no-reconall \
                              --participant-label sub-34 \
                               --fs-license-file /work/YOUR_USERNAME/license.txt