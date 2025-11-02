---

# **BTB3D – Encoder–Decoder Submodule**

This submodule provides the **3D encoder–decoder (Vision Tokenizer)** used in **BTB3D**, designed for compressing volumetric CT scans into discrete latent tokens and reconstructing them with high fidelity.

---

## 🧠 Overview

The encoder–decoder enables:

* Volumetric tokenization and reconstruction of CT scans
* Learnable quantization with **LFQ**
* Automatic voxel-spacing normalization and HU scaling
* Efficient multi-GPU training and inference (H100 cluster compatible)
* Dataset-wide batch encoding and reconstruction utilities

---

## 📁 Directory Structure

```
encoder-decoder/
│
├── train.py                  # Entry point for training the 3D tokenizer
├── inference.py              # Inference for image, video, or NIfTI volumes
├── run_whole_dataset.py      # Batch encoding + decoding for full datasets
│
├── configs/
│   ├── magvit2_3d_model_config.yaml
│   ├── magvit2_3d_model_config_8x8x8.yaml
│   ├── magvit2_3d_model_config_8x8x8_perceptual.yaml
│   ├── magvit2_3d_train_config.yaml
│   └── magvit2_3d_train_8x8x8_config_perceptual.yaml
└── README.md
```

---

## ⚙️ Environment Setup

Activate your Python environment:

```bash
module load python
conda activate magviot2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧩 Training

Train the 3D Vision Tokenizer using:

```bash
python train.py \
  --model_config_path configs/magvit2_3d_model_config_8x8x8_perceptual.yaml \
  --trainer_config_path configs/magvit2_3d_train_8x8x8_config_perceptual.yaml
```

Training supports distributed and mixed-precision execution:

```bash
accelerate launch --multi_gpu --mixed_precision bf16 train.py \
  --model_config_path ... \
  --trainer_config_path ...
```

Each configuration file specifies encoder/decoder layers, quantizer parameters, and training setup.

---

## 🧮 Encoding an Entire Dataset

The script `run_whole_dataset.py` processes a dataset folder (containing `.nii` or `.nii.gz` volumes), performs spacing-aware preprocessing, and generates both **quantized** and **decoded** outputs.

### Example Command

```bash
torchrun --nproc_per_node=4 run_whole_dataset.py \
  --modal nii \
  --ckpt-path exps/ckpts/iter_42000_fixed.ckpt \
  --config-path configs/magvit2_3d_model_config_8x8x8.yaml \
  -i /path/to/CT-dataset/valid/ \
  -o /path/to/output/quantized/
```

### Outputs

```
output/quantized/*.npz     → quantized token representations
output/encoded/*.npz       → encoded embeddings
output/*.gz                → reconstructed NIfTI volumes
```

**Important:**
➡️ You must update the CSV metadata path in the script before running:

```python
df = pd.read_csv("/path/to/your/valid_metadata.csv")
```

This path appears inside `run_whole_dataset.py` and must point to your dataset’s metadata file containing fields such as
`VolumeName`, `RescaleSlope`, `RescaleIntercept`, `XYSpacing`, and `ZSpacing`.

---

## 🔍 Inference (Single Volume or Image)

Run inference on a single case using `inference.py`.

### NIfTI Volume

```bash
python inference.py \
  --modal nii \
  --ckpt-path exps/ckpts/iter_42000_fixed.ckpt \
  --config-path configs/magvit2_3d_model_config_8x8x8.yaml \
  -i sample_volume.nii.gz \
  -o recon_output/
```

**Outputs:**

* 3D reconstructed `.nii.gz` files

**Note:**
➡️ Update the metadata CSV path in `inference.py` as well:

```python
df = pd.read_csv("/path/to/your/valid_metadata.csv")
```

This ensures that voxel rescaling and HU normalization use your dataset’s correct metadata.

---

## 🧰 Key Components

| Component              | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `VisionTokenizer`      | MAGViT-2 3D encoder–decoder backbone                             |
| `LFQ`                  | Learnable vector quantizer (discrete latent codes)               |
| `src.utils`            | Utilities for resizing, preprocessing, and configuration loading |
| `run_whole_dataset.py` | Large-scale encoding and reconstruction                          |
| `inference.py`         | Single-volume or image inference                                 |

---

## 🧪 Typical Workflow

1. **Train** tokenizer using `train.py`
2. **Generate tokens** for datasets with `run_whole_dataset.py`
3. **Visualize or reconstruct** volumes via `inference.py`

---

## 📜 Citation

If you use this submodule, please cite:

```
@article{hamamci2025btb3d,
  title={Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging},
  author={Ibrahim Ethem Hamamci and Sezgin Er and Suprosanna Shit and others},
  journal={arXiv preprint arXiv:2510.20639},
  year={2025}
}
```

---


## 🏥 Acknowledgements

Developed at the **Menze Lab**, University of Zurich,
in collaboration with **NVIDIA**, **Imperial College London**, and **FAU Erlangen–Nürnberg**,
using **H100 GPU clusters** on **NHR@FAU**.

---
