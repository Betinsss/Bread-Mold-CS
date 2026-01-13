import io
import os
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageDraw
import base64
import requests

ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', 'Sik6GXwtYzgOU5A2qtmY')
ROBOFLOW_MODEL_URL = os.environ.get('ROBOFLOW_MODEL_URL', 'https://detect.roboflow.com/final-mold-bread-2-yyc5q/1')

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["image"]
    img_bytes = file.read()

    # Call Roboflow API
    response = requests.post(
        ROBOFLOW_MODEL_URL,
        params={"api_key": ROBOFLOW_API_KEY, "confidence": 25},
        files={"file": img_bytes}
    )
    results = response.json()

    # Load image to draw
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    mold_area = 0
    bread_area = w * h

    detections = results.get('predictions', [])

    # Create a mask to accurately calculate mold coverage without overlapping areas
    mold_mask = Image.new('L', (w, h), 0)
    mask_draw = ImageDraw.Draw(mold_mask)

    # Process detections and handle overlapping boxes for the same object type
    bread_boxes = []
    mold_boxes = []
    
    for box in detections:
        cls_name = box['class']
        conf = box['confidence']
        x1 = int(box['x'] - box['width'] / 2)
        y1 = int(box['y'] - box['height'] / 2)
        x2 = int(box['x'] + box['width'] / 2)
        y2 = int(box['y'] + box['height'] / 2)

        if "mold" in cls_name.lower():
            mold_boxes.append((x1, y1, x2, y2, cls_name, conf))
        else:  # bread or other food items
            bread_boxes.append((x1, y1, x2, y2, cls_name, conf))

    # Draw bread boxes first
    for x1, y1, x2, y2, cls_name, conf in bread_boxes:
        color = (0, 120, 255)  # Blue for bread/food items
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Increase font size for better visibility of labels in bounding boxes
        try:
            from PIL import ImageFont
            # Use a larger font for better visibility
            font = ImageFont.truetype("arial.ttf", 16)  # Use a default system font with larger size
            draw.text((x1, y1 - 15), f"{cls_name} {conf*100:.1f}%", fill=color, font=font)
        except:
            # Fallback to default font if specific font is not available
            draw.text((x1, y1 - 15), f"{cls_name} {conf*100:.1f}%", fill=color)

    # Draw mold boxes second (on top)
    for x1, y1, x2, y2, cls_name, conf in mold_boxes:
        color = (255, 0, 0)  # Red for mold
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Increase font size for better visibility of labels in bounding boxes
        try:
            from PIL import ImageFont
            # Use a larger font for better visibility
            font = ImageFont.truetype("arial.ttf", 16)  # Use a default system font with larger size
            draw.text((x1, y1 - 15), f"{cls_name} {conf*100:.1f}%", fill=color, font=font)
        except:
            # Fallback to default font if specific font is not available
            draw.text((x1, y1 - 15), f"{cls_name} {conf*100:.1f}%", fill=color)

        # Fill the mold area in the mask to prevent double counting overlapping regions
        # Use a more robust approach to handle overlapping bounding boxes
        mask_draw.rectangle([x1, y1, x2, y2], fill=255)

    # Count the number of pixels in the mold mask to get accurate area
    # This ensures overlapping regions are only counted once
    mold_pixels = sum(mold_mask.getpixel((x, y)) > 0 for x in range(w) for y in range(h))
    mold_area = mold_pixels

    # Verify mold coverage calculation with additional validation
    # Calculate coverage based on the actual bread area detected, not just the whole image
    total_bread_area = 0
    for x1, y1, x2, y2, cls_name, conf in bread_boxes:
        total_bread_area += (x2 - x1) * (y2 - y1)
    
    # If no bread detected, use the whole image as bread area
    if total_bread_area == 0:
        total_bread_area = bread_area

    if mold_pixels > 0 and total_bread_area > 0:
        coverage_ratio = min(mold_pixels / total_bread_area, 1.0)  # Cap at 100%
    else:
        coverage_ratio = 0.0  # No mold detected or invalid dimensions

    # Extract bread and mold types
    bread_types = list(set([name for _, _, _, _, name, _ in bread_boxes])) if bread_boxes else ["Unknown"]
    mold_types = list(set([name for _, _, _, _, name, _ in mold_boxes])) if mold_boxes else ["None"]
    
    # Estimate storage time and bread age based on mold coverage
    if coverage_ratio == 0:
        risk = "None"
        action = "Safe to eat"
        verdict = "Healthy"
        storage_time = "0-3 days"
        bread_age = "Fresh"
    elif coverage_ratio < 0.1:
        risk = "Low"
        action = "Safe to remove moldy part carefully."
        verdict = "Healthy"
        storage_time = "3-5 days"
        bread_age = "Slightly aged"
    elif coverage_ratio < 0.3:
        risk = "Moderate"
        action = "Do not eat. Dispose bread safely."
        verdict = "Not Healthy"
        storage_time = "5-7 days"
        bread_age = "Old"
    else:
        risk = "Severe"
        action = "Highly contaminated. Dispose immediately."
        verdict = "Not Healthy"
        storage_time = "7+ days"
        bread_age = "Very old"

    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({
        "risk": risk,
        "coverage": round(coverage_ratio * 100, 2),
        "action": action,
        "verdict": verdict,
        "bread_type": ", ".join(bread_types),
        "mold_type": ", ".join(mold_types),
        "storage_time": storage_time,
        "bread_age": bread_age,
        "annotated": f"data:image/jpeg;base64,{encoded_img}"
    })


if __name__ == "__main__":
    app.run(debug=True)
