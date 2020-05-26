#!/bin/bash
#SBATCH --partition=titanx-short
#SBATCH --gres=gpu:2
#SBATCH --mem=16000

CUDA_VISIBLE_DEVICES=0,1 python train.py
