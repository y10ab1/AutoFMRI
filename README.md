# AutoFMRI: An Automated Two-stage Thresholding for fMRI Decoding

## Overview
Decoding models interpret neural activity to infer information in brain-computer interfaces. The challenge is the excess of features compared to the number of trials due to the high spatial resolution of fMRI. To address this, we introduce a two-stage thresholding technique that selects relevant voxels from the entire brain, improving decoding performance. This approach has shown to enhance regression performance in decoding musical pitch value, with a significant increase compared to restricting voxels to the auditory cortex alone. We also compare the performance of random forest and convolutional neural network decoders in this context.

## Environment Setup
**Note:** The `cuml` package requires CUDA version between 11.4 and 11.8. Make sure your CUDA version falls within this range before proceeding. We recommend using CUDA 11.7.


1. **Install Conda:**

   - Follow the official Conda installation instructions for your operating system.

2. **Create and activate the Conda environment:**
  
   - Run the following command to create the environment:
     ```
     conda env create -f environment.yml
     ```

   - Activate the Conda environment:
     ```
     conda activate fmri
     ```
3. **Install additional packages using pip:**

   - Run the following command to install additional packages using pip:
     ```
     pip install -r requirements.txt
     ```

## Usage

### Data Preparation

   - Run the following command to train the model:
     ```
     python train.py --data_dir /path/to/data
     ```
   - You can also specify the model type for each stage:
     ```
     python train.py --stage1_model rf --stage2_model cnn
     ```