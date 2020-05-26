#!/bin/bash
#SBATCH --partition=titanx-short
#SBATCH --gres=gpu:1
#SBATCH --mem=16000

CUDA_VISIBLE_DEVICES=0 python train.py
