# CACR-Net

Code for JBHI 2021 "Multi-Modality MR Image Synthesis via Confidence-Guided Aggregation and Cross-Modality Refinement"

Usage

    The original implementation of CACR-Net is Pytorch. The code has been tested in Linux.

    To run the code, you should first install dependencies:

    pip install fire

    Setup all parameters in config.py

    Put your data into ./data (Some samples from BraTs2018 have been stored out in this file)

    Train

    CUDA_VISIBLE_DEVICES=0,1 python main.py train --batch_size=8 --task_id=2 --gpu_id=[0,1]
The code of the CACR-Net is based on the Hi-Net [https://ieeexplore.ieee.org/abstract/document/9004544].

    
