# Geometric Gradient Rectification: Robust Prompt Tuning for Vision-Language Models

Official PyTorch implementation of **GGRP**.

## Setup

```bash
git clone https://github.com/BoyangGuo1789/GGRP.git
cd GGRP

conda create -y -n ggrp python=3.8
conda activate ggrp

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..
```

## Data

Prepare datasets with the same directory format as CoOp/Dassl.

```bash
export DATA_ROOT=/path/to/datasets
```

## Training

```bash
DATA_ROOT=/path/to/datasets \
GPU_ID=0 \
bash scripts/train.sh caltech101 100 ema 0.99 16 100
```

Arguments:
- `caltech101`: dataset name
- `100`: number of classes (manual input)
- `ema`: teacher mode (`freeze` or `ema`)
- `0.99`: teacher EMA momentum
- `16`: prompt length
- `100`: max epoch

Seed behavior follows CoOp style:
- `SEEDS=-1` (default): random mode
- `SEEDS>=0`: fixed seed

## Notes

- Local experiment scripts under `scripts/`.
- This code is built on top of CoOp, Dassl.pytorch, and CLIP.


