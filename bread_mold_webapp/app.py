import io
import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image, ImageDraw
import base64
import tempfile
import torch

# Handle PyTorch 2.6+ security changes for loading models
# Add safe globals for ultralytics models and their dependencies
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except ImportError:
    pass

try:
    # Add specific modules that may be needed by the model
    import ultralytics.nn.modules.conv
    import ultralytics.nn.modules.block
    import ultralytics.nn.modules.head
    # Add them to safe globals if they exist
    torch.serialization.add_safe_globals([
        ultralytics.nn.modules.conv.Conv,
        ultralytics.nn.modules.block.C2f,
        ultralytics.nn.modules.head.Detect,
        ultralytics.nn.modules.block.Bottleneck
    ])
except ImportError:
    pass

try:
    # Also add torch.nn modules that might be needed
    from torch.nn.modules.container import Sequential
    from torch.nn.modules.activation import SiLU, Sigmoid
    from torch.nn.modules.pooling import MaxPool2d
    from torch.nn.modules.linear import Linear
    from torch.nn.modules.normalization import BatchNorm2d
    torch.serialization.add_safe_globals([Sequential, SiLU, Sigmoid, MaxPool2d, Linear, BatchNorm2d])
except ImportError:
    pass

# Additional modules that might be needed
try:
    torch.serialization.add_safe_globals([
        torch.nn.modules.container.Sequential
    ])
except AttributeError:
    pass

# === Load local YOLO model (.pt file) ===
# ...existing code...
# === Load local YOLO model (.pt file) ===
# use model file located next to this script
MODEL_PATH = os.path.join(os.path.dirname(__file__), "my_model.pt")
model = YOLO(MODEL_PATH)
# ==============================================

# ==============================================

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["image"]
    img_bytes = file.read()

    # FIX: create temp file properly for Windows
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.write(img_bytes)
    temp.close()

    # Run YOLO prediction safely with lower confidence threshold to detect smaller mold areas
    results = model.predict(source=temp.name, conf=0.25)

    # Load image to draw
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    mold_area = 0
    bread_area = w * h

    detections = results[0].boxes

    # Create a mask to accurately calculate mold coverage without overlapping areas
    mold_mask = Image.new('L', (w, h), 0)
    mask_draw = ImageDraw.Draw(mold_mask)

    # Process detections and handle overlapping boxes for the same object type
    bread_boxes = []
    mold_boxes = []
    
    for box in detections:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to int for pixel operations

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

    # Cleanup temp file
    os.unlink(temp.name)

    if coverage_ratio == 0:
        risk = "None"
        action = "Safe to eat"
        verdict = "Healthy"
    elif coverage_ratio < 0.1:
        risk = "Low"
        action = "Safe to remove moldy part carefully."
        verdict = "Healthy"
    elif coverage_ratio < 0.3:
        risk = "Moderate"
        action = "Do not eat. Dispose bread safely."
        verdict = "Not Healthy"
    else:
        risk = "Severe"
        action = "Highly contaminated. Dispose immediately."
        verdict = "Not Healthy"

    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({
        "risk": risk,
        "coverage": round(coverage_ratio * 100, 2),
        "action": action,
        "verdict": verdict,
        "annotated": f"data:image/jpeg;base64,{encoded_img}"
    })


if __name__ == "__main__":
    app.run(debug=True)
