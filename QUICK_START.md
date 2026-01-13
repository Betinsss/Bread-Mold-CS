# ⚡ Quick Start Guide

## 🚀 Run in 3 Steps

### Step 1: Configure API Keys
```bash
# Copy the example file
copy .env.example .env

# Edit .env and add your Roboflow credentials:
# ROBOFLOW_API_KEY=your_key_here
# ROBOFLOW_MODEL_URL=your_url_here
```

### Step 2: Install & Start Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```
✅ Backend running at: **http://localhost:8000**

### Step 3: Open Web Interface
```bash
# Option A: Simple HTTP server
cd bread_mold_webapp
python -m http.server 8080
# Then open: http://localhost:8080

# Option B: Direct file access
# Just open: bread_mold_webapp/templates/index.html
```

---

## 🎯 That's It!

Upload a bread image and click "Analyze" to see:
- ✅ Detection Summary (Risk, Coverage, Types)
- ✅ Annotated Image + Verdict
- ✅ Result Breakdown (4 detailed cards)
- ✅ Mold Information (health risks & characteristics)
- ✅ Recommended Actions (immediate steps & storage tips)

---

## 📝 Requirements

- Python 3.8+
- Roboflow API account
- Modern web browser

---

## ❓ Issues?

See **HOW_TO_RUN.md** for detailed instructions and troubleshooting.
