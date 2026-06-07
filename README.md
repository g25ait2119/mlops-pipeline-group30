# End-to-End MLOps Pipeline — Image Classification

IIT Jodhpur · PGD AI · MLOps Group Assignment

Fine-tunes a compact image-classification model (TBD) on a
small public image dataset (TBD), tracks experiments on Weights & Biases, stores
the model on the Hugging Face Hub, containerises inference with Docker, and
automates linting + inference with GitHub Actions.

## Team
| Member             | Roll No. | Main responsibilities |
|--------------------|----------|-----------------------|
| Suresh Babu Gandla | `G25AIT2119` | Task 1 (repo), Task 7 (GitHub Actions) |
| Momin Mohd Asif Mohd Naeem               | `G25AIT2063` | Task 2 (data), Task 4 (training), Task 5 (HF push), Task 8 (W&B) |
| Vishnu Priya             | `G25AIT2128` | Task 6 (Docker) |
| All                |          | Task 3 (model selection) |

## Project structure
```
.
├── src/
│   ├── config.py            # all constants (model, dataset, labels, paths)
│   ├── data_preparation.py  # Task 2: inspect + clean images + id2label.json
│   ├── train.py             # Task 4: fine-tune + W&B logging
│   ├── evaluate.py          # test-set metrics
│   ├── inference.py         # Task 6/7: classify a single image
│   ├── metrics.py           # accuracy + weighted F1
│   └── utils.py             # seeding, secrets, device
├── .github/workflows/
│   ├── ci.yml               # Task 7.1: flake8 on push to develop
│   └── inference.yml        # Task 7.2: manual inference run (image URL input)
├── notebooks/
│   └── kaggle_train.py      # Task 4: copy/paste cells for Kaggle
├── Dockerfile               # Task 6
├── requirements.txt
├── id2label.json
├── LICENSE
└── .gitignore
```

> Model and dataset are being finalised by Asif. Constraints to keep marks safe:
> model under 200 MB and compatible with `AutoModelForImageClassification`
> (e.g. MobileNetV2, ResNet-18, ViT-tiny / DeiT-tiny); dataset small enough for
> Kaggle's free GPU (e.g. beans, Fashion-MNIST, or a CIFAR-10 subset).

## Setup
```bash
git clone https://github.com/g25ait2119/mlops-pipeline-group30.git
cd mlops-image-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## How to run each script (filled in by owners)
```bash
# Task 2 — prepare data + write id2label.json
python src/data_preparation.py

# Task 4 — train (set WANDB_API_KEY / HF_TOKEN in env first)
python src/train.py --version v1 ...
python src/train.py --version v2 ...

# Inference (image URL or local path)
IMAGE_URL="https://example.com/cat.jpg" python src/inference.py
```

## Docker
```bash
docker build --build-arg HF_MODEL_NAME=<user>/<model> -t mlops-pipeline-group30:latest .
docker run --rm -e IMAGE_URL="https://example.com/cat.jpg" mlops-pipeline-group30:latest
```

## Public links (fill in before submission)
- GitHub repo: `https://github.com/g25ait2119/mlops-pipeline-group30`
- Kaggle notebook v1: `<link>`
- Kaggle notebook v2: `<link>`
- Hugging Face model: `https://huggingface.co/<user>/<model>`
- Docker image: `https://hub.docker.com/r/<user>/mlops-pipeline-group30`
- W&B project: `https://wandb.ai/<entity>/mlops-pipeline-group30`
