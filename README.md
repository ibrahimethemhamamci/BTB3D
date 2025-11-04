# **BTB3D: Better Tokens for Better 3D [NeurIPS 2025]**

Official repository for
**“Better Tokens for Better 3D: Advancing Vision–Language Modeling in 3D Medical Imaging”**

**Ibrahim Ethem Hamamcı***, **Sezgin Er***, **Suprosanna Shit***, Hadrien Reynaud, Dong Yang, Pengfei Guo, Marc Edgar, Daguang Xu, Bernhard Kainz, Bjoern Menze

arXiv: [2510.20639](https://arxiv.org/abs/2510.20639), 2025

**🔗 Resources:**

* Model Weights: [**BTB3D on Hugging Face**](https://huggingface.co/Forithmuscom/BTB3D)
* Dataset: [**CT-RATE on Hugging Face**](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)

---

<p align="center">
  <img src="./figures/neurips_fig.png" alt="BTB3D Framework Overview" width="90%">
</p>

---

## Overview

**BTB3D** introduces a unified framework for **3D vision–language modeling in medical imaging**, combining **volumetric tokenization**, **report generation**, and **conditional 3D chest CT generation**.
It establishes a scalable bridge between radiological imaging and language through modular, reusable components designed for multimodal learning.

---

## Repository Structure

```
BTB3D/
│
├── encoder-decoder/          # 3D MAGViT tokenizer for volumetric representation
│
├── report-generation/        # LLaVA-based CT-CHAT model for report generation
│
├── ct-generation/            # Text-conditional CT generation with flow matching
│
├── figures/
│   └── neurips_fig.png
│
├── LICENSE
└── README.md
```

Each folder includes its own `README.md` detailing configuration, dependencies, and usage.

---

## Components

| Component             | Description                                                                                                                               |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Encoder–Decoder**   | 3D MAGViT-2–based tokenizer that compresses CT volumes into discrete latent codes (`.npz`), forming a foundation for downstream modeling. |
| **Report Generation** | LLaVA-based multimodal model (LLaMA-3.1-8B backbone) trained on (encoded-CT, report) pairs for radiology report generation.               |
| **CT Generation**     | Flow-matching–based text-conditional generator that reconstructs or synthesizes CT volumes directly from natural-language prompts.        |

---

## 🧩 Workflow Summary

1. **Encode 3D CT volumes** using the **Vision Tokenizer** (`encoder-decoder/`)
2. **Generate or fine-tune reports** using the **LLaVA-based model** (`report-generation/`)
3. **Synthesize CT volumes** from text prompts using the **Flow Matching generator** (`ct-generation/`)

---

## Citation

If you use this repository or any of its components, please cite:

```
@article{hamamci2025btb3d,
  title={Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging},
  author={Ibrahim Ethem Hamamci and Sezgin Er and Suprosanna Shit and Hadrien Reynaud and Dong Yang and Pengfei Guo and Marc Edgar and Daguang Xu and Bernhard Kainz and Bjoern Menze},
  journal={arXiv preprint arXiv:2510.20639},
  year={2025},
}
```

---
