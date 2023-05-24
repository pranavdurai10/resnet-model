# ResNet Model Library
 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

This is an un-official repository that contains multiple PyTorch implementations of the famous Image Classification Model, ResNet.

The ResNet model was first proposed in Deep Residual Learning for Image Recognition by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun.
Link to paper: https://arxiv.org/abs/1512.03385.

## Models

There are 7 implementations of ResNet:

1. ResNet18     [DONE]
2. ResNet34     [YET TO IMPLEMENT]
3. ResNet50     [YET TO IMPLEMENT]
4. ResNet101    [YET TO IMPLEMENT]
5. ResNet152    [YET TO IMPLEMENT]
6. ResNet164    [YET TO IMPLEMENT]
7. ResNet1202   [YET TO IMPLEMENT]

## Training

This repository contains a training script named as train.py in the root directory.

As default, it has been set to perform the following operations:
1. General Data Preprocessing: Image Augmentations
2. Load Dataset: CIFAR10 (default)
3. Define Dataloaders: For BatchSize, Shuffle, and Workers
4. Training Loop: 10 Epochs (default)
5. Validations Metrics: Display
6. Export Trained Model