import json
import os
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import ConvNextImageProcessor, AutoModelForImageClassification


MODEL_ID = os.getenv("HF_MODEL_NAME", "computervisionpro/convnextv2-real-fake")


def load_image(image_source: str) -> Image.Image:
    if image_source.startswith(("http://", "https://")):
        response = requests.get(image_source, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_source).convert("RGB")


def predict(image_source: str, model_id: str = MODEL_ID) -> dict:
    device = "cpu"
    hf_token = os.getenv("HF_TOKEN") or None

    processor = ConvNextImageProcessor.from_pretrained(model_id, token=hf_token)
    model = AutoModelForImageClassification.from_pretrained(model_id, token=hf_token)
    model.to(device)
    model.eval()

    image = load_image(image_source)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs).item())
    label = model.config.id2label.get(pred_id, str(pred_id))
    confidence = float(probs[pred_id].item())

    return {
        "image": image_source,
        "model": model_id,
        "prediction": label,
        "confidence": confidence,
        "probabilities": {
            model.config.id2label.get(i, str(i)): float(prob.item())
            for i, prob in enumerate(probs)
        },
    }


def main() -> None:
    image_source = os.getenv("IMAGE_URL") or os.getenv("IMAGE_PATH")
    if not image_source:
        raise ValueError("Set IMAGE_URL or IMAGE_PATH to run inference.")

    result = predict(image_source=image_source)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

# Made with Bob
