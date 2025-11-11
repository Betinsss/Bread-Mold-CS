import io
import os
from flask import Flask, render_template, request, jsonify
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import base64
import tempfile

# === Replace with your Roboflow info ===
API_KEY = "YQmgXwmCf6rUZEiZpi0CP"
PROJECT_NAME = "mold-bread-b03lw"
MODEL_VERSION = 2
# =======================================

app = Flask(__name__)

rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project(PROJECT_NAME)
model = project.version(MODEL_VERSION).model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["image"]
    img_bytes = file.read()

    # Save temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp:
        temp.write(img_bytes)
        temp.flush()
        result = model.predict(temp.name, confidence=40, overlap=30).json()

    # Load image to draw
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    mold_area = 0
    bread_area = w * h

    for obj in result["predictions"]:
        cls = obj["class"]
        conf = obj["confidence"]
        x, y, bw, bh = obj["x"], obj["y"], obj["width"], obj["height"]
        left = x - bw / 2
        top = y - bh / 2
        right = x + bw / 2
        bottom = y + bh / 2

        color = (255, 0, 0) if "mold" in cls.lower() else (0, 120, 255)
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.text((left, top - 10), f"{cls} {conf*100:.1f}%", fill=color)
        if "mold" in cls.lower():
            mold_area += bw * bh

    coverage_ratio = mold_area / bread_area
    if coverage_ratio < 0.1:
        risk = "Low"
        action = "Safe to remove moldy part carefully."
    elif coverage_ratio < 0.3:
        risk = "Moderate"
        action = "Do not eat. Dispose bread safely."
    else:
        risk = "Severe"
        action = "Highly contaminated. Dispose immediately."

    # Convert annotated image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({
        "risk": risk,
        "coverage": round(coverage_ratio * 100, 2),
        "action": action,
        "annotated": f"data:image/jpeg;base64,{encoded_img}"
    })


if __name__ == "__main__":
    app.run(debug=True)
