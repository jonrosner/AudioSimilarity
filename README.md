# Deep Learning for Determining Audio Similarity

This repository contains the thesis and code for my Master's Thesis "Deep Learning for Determining Audio Similarity".

## Abstract

We propose a deep learning framework for similarity learning of audio representations. It is inspired by the most recent successes in self-supervised representation learning in the domain of image data. Our framework transfers those successes to the domain of audio data. With our framework we show that (1) self-supervised contrastive learning can successfully learn robust audio representations without the need for labels, (2) the learned representations can be applied to other audio-based downstream tasks using transfer learning and (3) we show that our approach outperforms recently published results in the area of few-shot classification for audio data. We further describe the preliminary knowledge in signal processing and deep learning required to understand the inner workings of our proposed framework and investigate the most recent and most important related work in this field of research.

## Thesis

The built pdf thesis can be found in `/thesis/build/main.pdf`.

## Run Code Locally

Follow these steps to run the conducted experiments locally (python3.x must be installed):

1. Initiate a virtual environment using `python -m venv venv` and source it with `source venv/bin/activate`.
2. Install all dependencies with `pip install -r requirements.txt`.
3. Copy the desired dataset and/or saved models to a desired localtion
4. Configure the hyperparameters in `experiments.py`.
5. Run `python experiments.py`

## Datasets

- The VoxCeleb dataset (and split file) can be downloaded from http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
- The British Birdsong dataset can be downlaoded from https://www.kaggle.com/rtatman/british-birdsong-dataset
