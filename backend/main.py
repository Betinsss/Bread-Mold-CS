from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw
import io
import base64
import requests
import os
from datetime import datetime
from typing import List, Optional
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Bread Mold Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
ROBOFLOW_MODEL_URL = os.getenv('ROBOFLOW_MODEL_URL')

if not ROBOFLOW_API_KEY or not ROBOFLOW_MODEL_URL:
    print('WARNING: ROBOFLOW_API_KEY and ROBOFLOW_MODEL_URL not found in environment')
    print('Please create a .env file with your credentials')
    raise ValueError('ROBOFLOW_API_KEY and ROBOFLOW_MODEL_URL must be set in .env file')
HISTORY_FILE = "analysis_history.json"

class AnalysisResult(BaseModel):
    id: str
    timestamp: str
    risk: str
    coverage: float
    action: str
    verdict: str
    bread_type: str
    mold_type: str
    storage_time: str
    bread_age: str
    age_days: int
    annotated: str
    annotated_no_labels: str
    detections_count: dict

def load_history() -> List[dict]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_to_history(result: dict):
    history = load_history()
    history.append(result)
    if len(history) > 100:
        history = history[-100:]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

@app.get("/")
def root():
    return {"message": "Bread Mold Detection API", "version": "2.0"}

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()
    
    response = requests.post(
        ROBOFLOW_MODEL_URL,
        params={"api_key": ROBOFLOW_API_KEY, "confidence": 25},
        files={"file": img_bytes}
    )
    results = response.json()
    
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    
    detections = results.get('predictions', [])
    mold_mask = Image.new('L', (w, h), 0)
    mask_draw = ImageDraw.Draw(mold_mask)
    
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
        else:
            bread_boxes.append((x1, y1, x2, y2, cls_name, conf))
    
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = None
    
    for x1, y1, x2, y2, cls_name, conf in bread_boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(139, 90, 60), width=3)
        if font:
            draw.text((x1, y1 - 30), f"{cls_name} {conf*100:.1f}%", fill=(139, 90, 60), font=font)
        else:
            draw.text((x1, y1 - 15), f"{cls_name} {conf*100:.1f}%", fill=(139, 90, 60))
    
    for x1, y1, x2, y2, cls_name, conf in mold_boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(180, 82, 45), width=3)
        if font:
            draw.text((x1, y1 - 30), f"{cls_name} {conf*100:.1f}%", fill=(180, 82, 45), font=font)
        else:
            draw.text((x1, y1 - 15), f"{cls_name} {conf*100:.1f}%", fill=(180, 82, 45))
        mask_draw.rectangle([x1, y1, x2, y2], fill=255)
    
    mold_pixels = sum(1 for x in range(w) for y in range(h) if mold_mask.getpixel((x, y)) > 0)
    total_bread_area = sum((x2-x1)*(y2-y1) for x1,y1,x2,y2,_,_ in bread_boxes) or w*h
    coverage_ratio = min(mold_pixels / total_bread_area, 1.0) if total_bread_area > 0 else 0.0
    
    bread_types = list(set([name.replace('_', ' ') for _,_,_,_,name,_ in bread_boxes])) if bread_boxes else (["Flat bread"] if mold_boxes else ["Unknown"])
    mold_types = list(set([name.replace('_', ' ') for _,_,_,_,name,_ in mold_boxes])) or ["None"]
    
    bread_recommendations = {
        "Flat bread": "Store in airtight container at room temperature for 2-3 days or refrigerate for up to 1 week",
        "White bread": "Keep in original packaging at room temperature for 5-7 days or freeze for up to 3 months",
        "Whole wheat": "Store in cool, dry place for 3-5 days or refrigerate to extend freshness",
        "Sourdough": "Keep cut-side down on cutting board or in paper bag for 2-3 days"
    }
    
    mold_info_map = {
        "aspergillus": "Aspergillus is a black or dark-colored mold that can produce harmful mycotoxins. It appears as black, brown, or greenish-black spots on bread. This mold poses a very high risk as it can cause respiratory issues, allergic reactions, and in severe cases, aspergillosis. It thrives in warm, humid conditions and can spread rapidly through bread's porous structure.",
        "cladosporium": "Cladosporium is an olive-green to brown or black mold commonly found on bread and other food items. It appears as dark green to black velvety patches. While it poses a moderate to high risk, it's one of the most common indoor molds. It can trigger allergies, asthma symptoms, and respiratory issues, especially in sensitive individuals. This mold grows well in both warm and cool conditions.",
        "penicillium": "Penicillium is a blue-green mold and the most common bread mold species. It appears as blue, green, or white fuzzy growth on the bread surface. This mold poses a high risk as it spreads rapidly throughout the bread's internal structure. While generally not as toxic as some other molds, it can cause allergic reactions and respiratory problems. The spores can easily become airborne and contaminate other food items.",
        "rhizopus": "Rhizopus, also known as black bread mold, grows quickly in warm, moist conditions. It initially appears as white fuzzy growth that turns black as spores develop. This mold poses a high risk due to its rapid growth rate. It can cause infections (mucormycosis) in immunocompromised individuals and produce toxins. The mold spreads through stolons (horizontal stems) and can quickly contaminate entire loaves."
    }
    
    if coverage_ratio == 0:
        risk, action, verdict, storage_time, bread_age, age_days = "None", "• Bread appears fresh and safe to consume\n• Store in a cool, dry place away from direct sunlight\n• Consider refrigeration in humid climates to prevent mold growth\n• Use within recommended timeframe for best quality", "Healthy", "0-3 days", "Fresh", 1
    elif coverage_ratio < 0.1:
        risk, action, verdict, storage_time, bread_age, age_days = "Low", "• Minor mold detected on surface\n• Recommended to discard as spores may have spread internally\n• If consuming, cut away moldy section with 1-inch margin\n• Toast thoroughly and monitor for adverse reactions", "Healthy", "3-5 days", "Slightly aged", 4
    elif coverage_ratio < 0.3:
        risk, action, verdict, storage_time, bread_age, age_days = "Moderate", "• Significant mold contamination detected\n• Do not consume - mold has penetrated deep into the product\n• Dispose in sealed plastic bag to prevent spore spread\n• Clean storage area with diluted bleach solution\n• Check other bread products nearby for contamination", "Not Healthy", "5-7 days", "Old", 6
    else:
        risk, action, verdict, storage_time, bread_age, age_days = "Severe", "• Extensive mold contamination throughout bread\n• Serious health risks: allergic reactions and respiratory issues\n• Immediately dispose in sealed container\n• Thoroughly clean and disinfect storage area and surfaces\n• Inspect all nearby food items for cross-contamination\n• Ventilate area to remove airborne spores", "Not Healthy", "7+ days", "Very old", 10
    
    mold_rec = ""
    if mold_types[0] != "None":
        mold_key = mold_types[0].lower().replace('_', ' ').replace('mold ', '')
        mold_rec = next((v for k, v in mold_info_map.items() if k in mold_key), "")
    
    max_size = 800
    if w > max_size or h > max_size:
        ratio = min(max_size/w, max_size/h)
        new_size = (int(w*ratio), int(h*ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    image_no_labels = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if w > max_size or h > max_size:
        image_no_labels = image_no_labels.resize(new_size, Image.Resampling.LANCZOS)
    draw_no_labels = ImageDraw.Draw(image_no_labels)
    
    for x1, y1, x2, y2, cls_name, conf in bread_boxes:
        draw_no_labels.rectangle([x1, y1, x2, y2], outline=(139, 90, 60), width=3)
    
    for x1, y1, x2, y2, cls_name, conf in mold_boxes:
        draw_no_labels.rectangle([x1, y1, x2, y2], outline=(180, 82, 45), width=3)
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    buffer_no_labels = io.BytesIO()
    image_no_labels.save(buffer_no_labels, format="JPEG")
    encoded_img_no_labels = base64.b64encode(buffer_no_labels.getvalue()).decode("utf-8")
    
    result = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "risk": risk,
        "coverage": round(coverage_ratio * 100, 2),
        "action": action,
        "verdict": verdict,
        "bread_type": ", ".join(bread_types),
        "mold_type": ", ".join(mold_types),
        "storage_time": storage_time,
        "bread_age": bread_age,
        "age_days": age_days,
        "annotated": f"data:image/jpeg;base64,{encoded_img}",
        "annotated_no_labels": f"data:image/jpeg;base64,{encoded_img_no_labels}",
        "detections_count": {"bread": len(bread_boxes) if bread_boxes else (1 if mold_boxes else 0), "mold": len(mold_boxes)},
        "mold_information": mold_rec
    }
    
    save_to_history(result)
    return result

@app.get("/api/history")
def get_history():
    return load_history()

@app.get("/api/analytics")
def get_analytics():
    history = load_history()
    if not history:
        return {
            "total_scans": 0, "healthy": 0, "unhealthy": 0, "avg_coverage": 0,
            "risk_distribution": {}, "bread_types": {}, "mold_types": {}, "age_distribution": {}
        }
    
    total = len(history)
    healthy = sum(1 for h in history if h['verdict'] == 'Healthy')
    unhealthy = total - healthy
    avg_coverage = sum(h['coverage'] for h in history) / total
    
    risk_dist = {}
    bread_types = {}
    mold_types = {}
    age_dist = {}
    
    for h in history:
        risk_dist[h['risk']] = risk_dist.get(h['risk'], 0) + 1
        
        for bread in h['bread_type'].split(', '):
            bread_types[bread] = bread_types.get(bread, 0) + 1
        
        for mold in h['mold_type'].split(', '):
            mold_types[mold] = mold_types.get(mold, 0) + 1
        
        age_dist[h['bread_age']] = age_dist.get(h['bread_age'], 0) + 1
    
    return {
        "total_scans": total,
        "healthy": healthy,
        "unhealthy": unhealthy,
        "avg_coverage": round(avg_coverage, 2),
        "risk_distribution": risk_dist,
        "bread_types": bread_types,
        "mold_types": mold_types,
        "age_distribution": age_dist
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
