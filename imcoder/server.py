from flask import Flask, request
from PIL import Image
import io

import numpy as np
import torch

from imcoder.encoders.models import get_model

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


def remove_alpha(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image)
        image = image.convert("RGB")
    return image


def encode(image: Image.Image, model_name: str) -> np.ndarray:
    # make sure the image is in RGB mode
    image = remove_alpha(image)

    # load the model and preprocess the image
    model, preprocess = get_model(model_name)
    model.to(device)
    model.eval()  # set the model into evaluation mode

    # encode the image
    with torch.no_grad():
        pixels = preprocess(image)
        pixels = pixels.to(device)
        pixels = pixels.unsqueeze(0)
        encoded = model(pixels)

    # get the image back to the cpu and numpy
    encoded = encoded.detach().cpu().numpy()
    return encoded


@app.route("/encode", methods=["POST"])
def encode_file():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]

    image_content = file.read()

    try:
        # check if the image is valid
        with Image.open(io.BytesIO(image_content)) as image:
            image.verify()
    except (IOError, SyntaxError):
        return "Invalid image file", 400

    with Image.open(io.BytesIO(image_content)) as image:
        model_name = request.form["model"]
        encoded = encode(image, model_name)
        encoded = encoded.tolist()
        return {"encoded": encoded}
