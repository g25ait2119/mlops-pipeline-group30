# End-to-End MLOps Pipeline — Image Classification

IIT Jodhpur · PGD AI · MLOps Group Assignment

Fine-tunes a compact image-classification model  on a
small public image dataset, tracks experiments on Weights & Biases, stores
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
│   ├── datasplit.py                
│   ├── inference.py                # classify a single image
├── .github/workflows/
│   ├── ci.yml                      # flake8 on push to develop
│   └── inference.yml               # manual inference run (image URL input)
├── notebooks/
│   └── group30-mlops-a3.ipynb      # copy/paste cells for Kaggle
├── Dockerfile                      
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
- Kaggle notebook: `https://www.kaggle.com/code/computervisionpro/group30-mlops-a3`
- Kaggle Dataset: `https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images`
- Hugging Face model(Base Model): `https://huggingface.co/facebook/convnextv2-tiny-1k-224`
- Hugging Face model(Hugging Face): `https://huggingface.co/computervisionpro/convnextv2-real-fake`
- Docker image: `https://hub.docker.com/r/sureshbabugandla1/mlops-group30-inference`
- W&B project: `https://wandb.ai/computervisionpro-na/mlops-assignment3`
