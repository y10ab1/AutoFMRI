# AutoFMRI: An Automated Two-stage Thresholding for fMRI Decoding

## Overview
This project aims to develop a system for simulating different fMRI parcellations and analyzing them using various strategies. The goal is to explore the performance of different packages, such as sklearn and cuML, on both CPU and GPU platforms. Additionally, the project will investigate methods like parallelization to enhance the efficiency of the time-consuming analysis process.

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