# ===========================================================================
# Dockerfile (Task 6) -- PLACEHOLDER
# Owner: Vishnu. Implement once Asif's image model is on the Hugging Face Hub.
#
# Target design (image classification):
#   - FROM python:3.11-slim
#   - ARG HF_MODEL_NAME=<user>/<image-model>   (sensible default)
#   - install requirements (transformers, torch, torchvision, Pillow, ...)
#   - COPY src/ and id2label.json
#   - run as non-root user
#   - CMD ["python", "src/inference.py]   # reads IMAGE_URL from env
#
# Build : docker build --build-arg HF_MODEL_NAME=<user>/<model> -t mlops-pipeline-group30:latest .
# Test  : docker run --rm -e IMAGE_URL="https://.../cat.jpg" mlops-pipeline-group30:latest
# Push  : docker push <user>/mlops-pipeline-group30:latest
# ===========================================================================
