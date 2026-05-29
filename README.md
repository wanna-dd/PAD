# PAD: Adaptive Feedback Cross-loop Self-KD for Robust Few-shot Spectral Learning

<p align="center">
  <a href="https://doi.org/10.1088/1361-6501/aded2a">
    <img src="https://img.shields.io/badge/Paper-Measurement%20Science%20and%20Technology-blue">
  </a>
  <img src="https://img.shields.io/badge/Python-3.8.13-green">
  <img src="https://img.shields.io/badge/PyTorch-2.3.1-orange">
  <img src="https://img.shields.io/badge/CUDA-12.0.151-brightgreen">
  <img src="https://img.shields.io/badge/Task-Few--shot%20Spectral%20Learning-purple">
</p>

<p align="center">
  <b>Official PyTorch implementation of PAD</b><br>
  Adaptive Feedback Cross-loop for Preserving and Robust Spectral Information Optimization without Spectral Processing in Few-shot Learning
</p>

---

## 🔥 Highlights

**PAD** is a spectral representation learning framework designed for robust few-shot classification of one-dimensional spectral signals, including **UV-vis**, **Raman**, and **infrared spectra**.

Unlike conventional pipelines that heavily rely on handcrafted spectral preprocessing, PAD learns to preserve and optimize discriminative spectral information through an **adaptive feedback cross-loop self-knowledge distillation strategy**.

Key features:

- **Spectral-preserving learning** without complex handcrafted preprocessing.
- **Odd-even spectral decoupling** for complementary spectral information extraction.
- **Adaptive cross-loop self-knowledge distillation** to improve representation robustness.
- **Few-shot friendly training strategy** for limited-sample spectral classification.
- Compatible with multiple one-dimensional deep learning backbones, including **ResNet**, **DRSN**, **LSTM**, and **MobileNetV3**.

---

## 🧠 Method Overview

PAD decomposes the original one-dimensional spectrum into two complementary spectral views:

- an odd-index spectral branch;
- an even-index spectral branch.

These two branches are used to construct an adaptive feedback cross-loop, where the model learns from both current and delayed spectral representations. Through self-knowledge distillation, PAD encourages the two branches to exchange reliable spectral information while preserving discriminative spectral patterns.

The core idea is to improve model robustness and generalization when only limited labeled spectral samples are available.

---

## 📌 Paper

This repository provides the official implementation of the following paper:

> **Adaptive feedback cross-loop for preserving and robust spectral information optimization without spectral processing in few-shot learning**  
> Yuduan Lin, Yalu Cai, Haotian Chen, Yitao Cai, Zhibiao Lin, Honghao Cai, Hui Ni  
> *Measurement Science and Technology*, 2025, 36, 075503.  
> DOI: [10.1088/1361-6501/aded2a](https://doi.org/10.1088/1361-6501/aded2a)

---

## 📁 Repository Structure

```text
PAD/
├── models/
│   ├── DRSN.py              # Deep residual shrinkage network backbone
│   ├── LSTM.py              # Recurrent spectral sequence model
│   ├── MobileNetV3.py       # Lightweight convolutional backbone
│   ├── ResNet.py            # Residual convolutional backbone
│   └── __init__.py          # Model loader
├── Datasets.py              # Dataset loading and preprocessing
├── Train_PAD.py             # Training script for PAD
├── Test.py                  # Testing and checkpoint evaluation
├── Utils.py                 # Logging and utility functions
├── Visualization.py         # GradCAM
└── README.md
