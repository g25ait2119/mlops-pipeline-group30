

import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "computervisionpro/convnextv2-real-fake"


def predict(image_path, model_id=MODEL_ID):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    # hf_token = os.getenv("HF_TOKEN") or None

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs).item())
    label = model.config.id2label.get(pred_id, str(pred_id))
    confidence = float(probs[pred_id].item())

    return {
        "image": image_path,
        "model": model_id,
        "prediction": label,
        "confidence": confidence,
        "probabilities": {
            model.config.id2label.get(i, str(i)): float(prob.item())
            for i, prob in enumerate(probs)
        },
    }


result = predict("./dataset/test/fake/fake_1006.jpg")
print()
print(result)