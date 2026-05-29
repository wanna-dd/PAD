# PAD: Adaptive Feedback Cross-loop Self-KD for Robust Few-shot Spectral Learning

<p align="center">
  <a href="https://doi.org/10.1088/1361-6501/aded2a">
    <img src="https://img.shields.io/badge/Paper-Measurement%20Science%20and%20Technology-blue">
    <img src="https://img.shields.io/badge/Task-Few--shot%20Spectral%20Learning-purple">
</p>

This repository is adapted from the codebase used to produce the results in the paper **Adaptive Feedback Cross-loop for Preserving and Robust Spectral Information Optimization without 
Spectral Processing in Few-shot Learning** published in **Measurement Science and Technology (2025)** at https://doi.org/10.1088/1361-6501/aded2a.

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

## ⚙️ Requirements
The code in this repo has been tested with the following software versions:

**Python** 3.8.13

**PyTorch** 2.3.1

**Cuda** 12.0.151

**Pycharm** 2024.1.4

We recommend using the Anaconda Python distribution, which is available for Windows, MacOS, and Linux.

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
├── Datasets.py              # Dataset loading
├── Train_PAD.py             # Training script for PAD
├── Test.py                  # Testing evaluation
├── Utils.py                 # Logging and utility functions
├── Visualization.py         # GradCAM-based visualization
└── README.md
```

---

## 📖 Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@article{Lin2025AdaptiveFC,
  title={Adaptive feedback cross-loop for preserving and robust spectral information optimization without spectral processing in few-shot learning},
  author={Yuduan Lin and Yalu Cai and Haotian Chen and Yitao Cai and Zhibiao Lin and Honghao Cai and Hui Ni},
  journal={Measurement Science and Technology},
  year={2025},
  volume={36},
  url={https://api.semanticscholar.org/CorpusID:280124913}
}
```
---

## 📄 License

This project is released for academic research purposes. Please cite the paper if you use this repository in your research.

For commercial use or redistribution, please contact the authors. The public Raman spectral dadaset for bacteria task in the experiment are available at https://github.com/csho33/bacteria-ID.
