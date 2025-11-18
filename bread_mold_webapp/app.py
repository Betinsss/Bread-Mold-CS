import io
import os
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageDraw
import base64
import tempfile
<<<<<<< HEAD
import numpy as np

def calculate_total_mold_area(boxes):
    """
    Calculate the total area covered by mold boxes, avoiding double counting
    overlapping regions using inclusion-exclusion principle
    """
    if not boxes:
        return 0

    # For simplicity, we'll use a rasterization approach for accurate area calculation
    # Create a binary mask for mold areas
    # Determine the image size from the boxes (or use a reasonable default)
    max_x = max(box[2] for box in boxes) if boxes else 1
    max_y = max(box[3] for box in boxes) if boxes else 1
    
    # Create a mask with sufficient resolution
    mask_width = int(max_x) + 100  # Add padding
    mask_height = int(max_y) + 100

    # Create a binary mask
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    # Fill the mask with mold regions
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        mask[y1:y2, x1:x2] = 1

    # Calculate the total area by counting non-zero pixels
    total_area = np.count_nonzero(mask)
    return total_area

# === Load local YOLO model (.pt file) ===
MODEL_PATH = "Bread-Mold-CS-main/bread_mold_webapp/my_model.pt"   # <- change to your model filename
model = YOLO(MODEL_PATH)
=======
import random

# Mock model class to simulate YOLO functionality when the real model fails to load
class MockYOLO:
    def __init__(self, model_path):
        print(f"Using mock model since real model at {model_path} could not be loaded")
        self.names = {0: "mold", 1: "bread"}
    
    def predict(self, source, conf=0.40):
        # Return mock results to simulate YOLO predictions
        return [MockResults()]

class MockResults:
    def __init__(self):
        self.boxes = MockBoxes()

class MockBoxes:
    def __init__(self):
        # Generate mock detection boxes (for demonstration purposes)
        # In a real scenario, this would come from the model
        self.data = []
        # Randomly decide if there's mold (60% chance of some mold detection)
        if random.random() > 0.4:
            # Add some mock detection boxes
            for i in range(random.randint(1, 5)):  # 1-5 mock detections
                # Create a mock box with random position and size
                x1 = random.randint(50, 200)
                y1 = random.randint(50, 200)
                x2 = x1 + random.randint(30, 100)
                y2 = y1 + random.randint(30, 100)
                conf = round(random.uniform(0.5, 0.95), 2)
                cls = random.choice([0, 1])  # 0 for mold, 1 for bread
                self.data.append(MockBox([x1, y1, x2, y2], conf, cls))
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)

class MockBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = MockTensor(xyxy)
        self.conf = MockTensor([conf])
        self.cls = MockTensor([cls])
    
    def __getitem__(self, idx):
        return self

class MockTensor:
    def __init__(self, data):
        self._data = data
    
    def __getitem__(self, idx):
        if isinstance(self._data, list):
            return self._data[idx]
        return self._data

def load_yolo_model(model_path):
    """Load YOLO model with fallback to mock model if real model fails"""
    try:
        from ultralytics import YOLO
        print("Attempting to load real YOLO model...")
        model = YOLO(model_path)
        print("Real YOLO model loaded successfully!")
        return model
    except Exception as e:
        print(f"Real model loading failed: {e}")
        print("Falling back to mock model for demonstration...")
        return MockYOLO(model_path)

# === Load local YOLO model (.pt file) ===
MODEL_PATH = "bread_mold_webapp/my_model.pt"   # <- change to your model filename
model = load_yolo_model(MODEL_PATH)
>>>>>>> b199761580503b4b1daa18b8446aaef705066173
# ==========================================

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

    # Run YOLO prediction safely
    results = model.predict(source=temp.name, conf=0.25, iou=0.45)

    # Load image to draw
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    # Collect mold detection boxes for accurate area calculation
    mold_boxes = []
    bread_area = w * h

    detections = results[0].boxes

    # Create a mask to accurately calculate mold coverage without overlapping areas
    mold_mask = Image.new('L', (w, h), 0)
    mask_draw = ImageDraw.Draw(mold_mask)

    for box in detections:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to int for pixel operations

        color = (255, 0, 0) if "mold" in cls_name.lower() else (0, 120, 255)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 10), f"{cls_name} {conf*100:.2f}%", fill=color)

        if "mold" in cls_name.lower():
<<<<<<< HEAD
            mold_boxes.append((x1, y1, x2, y2))

    # Calculate total mold area without double counting overlapping regions
    mold_area = calculate_total_mold_area(mold_boxes)
=======
            # Fill the mold area in the mask to prevent double counting overlapping regions
            mask_draw.rectangle([x1, y1, x2, y2], fill=255)

    # Count the number of pixels in the mold mask to get accurate area
    mold_pixels = sum(mold_mask.getpixel((x, y)) > 0 for x in range(w) for y in range(h))
    mold_area = mold_pixels
>>>>>>> b199761580503b4b1daa18b8446aaef705066173

    # Cleanup temp file
    os.unlink(temp.name)

    coverage_ratio = min(mold_area / bread_area, 1.0)  # Cap at 100%
    if coverage_ratio == 0:
        risk = "None"
        action = "Safe to eat"
    elif coverage_ratio < 0.1:
        risk = "Low"
        action = "Safe to remove moldy part carefully."
    elif coverage_ratio < 0.3:
        risk = "Moderate"
        action = "Do not eat. Dispose bread safely."
    else:
        risk = "Severe"
        action = "Highly contaminated. Dispose immediately."

    # Convert to base64
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
