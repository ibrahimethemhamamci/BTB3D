---

# **BTB3D – Report Generation Submodule**

This submodule provides the **report generation component** of **BTB3D**, built upon the **LLaVA (Large Language and Vision Assistant)** architecture.
It fine-tunes a **LLaMA-3.1-8B-Instruct** backbone with encoded 3D CT embeddings to generate structured radiology reports.
The model learns to align volumetric image features with textual findings using multimodal instruction tuning.

---

## 🧠 Overview

This module connects **3D encoded CT volumes** (from the encoder–decoder submodule) with **radiology reports** using the LLaVA framework.
It supports large-scale multimodal fine-tuning and multi-GPU inference for automated report generation.

Key features:

* LoRA-based fine-tuning of **LLaMA-3.1-8B**
* **Vision encoder:** CLIP ViT-L/14-336
* **Distributed training:** DeepSpeed ZeRO-2 + MPI
* **BF16 precision** for H100 GPUs
* Multi-GPU inference with multiprocessing

---

## 📁 Directory Structure

```
report-generation/
│
├── setup.py                          # Installation file
├── llava/
│   ├── train/train_mem.py            # Main LLaVA training script
│   └── serve/multigpu_llava_eval.py  # Multi-GPU inference script
└── README.md
```

---

## ⚙️ Installation

From the `report-generation` directory:

```bash
pip install -e .
```

Activate your environment (for example):

```bash
conda activate btb3d-llava
```

**Dependencies:**

```bash
torch torchvision deepspeed transformers accelerate sentencepiece pandas tqdm nibabel
```

---

## 🧩 Training

Training is performed using **LoRA fine-tuning** on a distributed H100 cluster.
While the training launcher is not included in the repository, a typical **SLURM setup** is shown below for reference.

### Example SLURM Script (for documentation)

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=llava_mem_%j.log
#SBATCH --error=llava_mem_%j.err

module load openmpi/5.0.5-gcc14.2.0-cuda
module load cuda/12.6.2
module load cudnn/9.2.0.82-12
source activate btb3d-llava


mpirun -np 8 \
  python ./llava/train/train_mem.py \
  --lora_enable True \
  --lora_r 128 \
  --lora_alpha 256 \
  --mm_projector_lr 2e-5 \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --version llama3_1 \
  --data_path /path/to/filtered_report_generation.json \
  --image_folder /path/to/encoded/ \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type attn_pool+mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --bf16 True \
  --output_dir ./llava_llama_lora_finetune_vqa \
  --num_train_epochs 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --save_strategy steps \
  --save_steps 2000 \
  --lr_scheduler_type cosine \
  --deepspeed scripts/zero2.json \
  --report_to none
```

### Key Settings

* **Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
* **Vision tower:** `openai/clip-vit-large-patch14-336`
* **Dataset:** `/path/to/filtered_report_generation.json`
* **Encoded CT features:** `/path/to/encoded/`
* **Output directory:** `./llava_llama_lora_finetune_vqa`
* **LoRA parameters:** `r=128`, `alpha=256`
* **Precision:** `bf16`
* **Epochs:** 100

---

## 🔍 Inference

Inference is performed with:

```
llava/serve/multigpu_llava_eval.py
```

This script runs **multi-GPU parallel inference** on pre-encoded CT embeddings.

### Example Command

```bash
python llava/serve/multigpu_llava_eval.py \
  --model-path ./llava_llama_lora_finetune_vqa/checkpoint-40000 \
  --model-base meta-llama/Meta-Llama-3.1-8B-Instruct \
  --embedding_path /path/to/encoded/ \
  --json_path /path/to/filtered_report_generation.json \
  --total_nodes 1 \
  --num_gpus 4
```

Each GPU processes a subset of CT embeddings and produces:

```
*_vqa.json → Generated reports (question–answer pairs)
```

---

## ⚠️ Important Configuration

Before running training or inference, update the dataset and embedding paths:

**For training:**

```
--data_path /path/to/filtered_report_generation.json
--image_folder /path/to/encoded/
```

**For inference:**

```python
args.embedding_path = "/path/to/encoded/"
args.json_path = "/path/to/vqa_reports.json"
```

These must match your dataset and the output directory from the encoder–decoder module.

---

## 🧰 Key Components

| Component                         | Description                                                     |
| --------------------------------- | --------------------------------------------------------------- |
| `train_mem.py`                    | Main training loop for multimodal instruction fine-tuning       |
| `multigpu_llava_eval.py`          | Distributed multi-GPU inference and report generation           |
| `setup.py`                        | Package installation entry point                                |
| `filtered_report_generation.json` | Radiology report dataset in LLaVA instruction format            |
| `encoded/`                        | Directory containing `.npz` embeddings from the encoder–decoder |

---

## 🧪 Typical Workflow

1. **Generate CT embeddings** using the encoder–decoder submodule
2. **Fine-tune the LLaVA model** with `(embedding + report)` pairs
3. **Run inference** with `multigpu_llava_eval.py` for automatic report generation
4. **Evaluate** generated reports using BLEU, ROUGE, or RadGraph metrics

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
in collaboration with **NVIDIA**, **Imperial College London**, and **FAU Erlangen-Nürnberg**,
using **H100 GPU clusters** on **NHR@FAU**.
