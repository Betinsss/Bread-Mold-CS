# How to Run the Bread Mold Detection System

## Prerequisites
- Python 3.8+
- Node.js 16+
- npm

## Quick Start

### 1. Backend Setup (FastAPI)
```bash
cd backend
pip install fastapi uvicorn pillow requests python-dotenv
python main.py
```
Backend runs on: http://localhost:8000

### 2. Frontend Setup (Node.js)

**Option A: Use Command Prompt (Recommended)**
```cmd
cd frontend
npm install
npm start
```

**Option B: Fix PowerShell Execution Policy**
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then run:
cd frontend
npm install
npm start
```

**Option C: Bypass PowerShell Policy**
```powershell
powershell -ExecutionPolicy Bypass -Command "cd frontend; npm install; npm start"
```

Frontend runs on: http://localhost:3000

### 3. Access the Application
Open your browser and go to: http://localhost:3000

## Alternative: Flask App (Simpler Setup)
If you prefer the Flask version:
```bash
cd bread_mold_webapp
pip install flask pillow requests
python app.py
```
Flask app runs on: http://localhost:5000

## Environment Setup
The `.env` file has been created with your API credentials. If you need to update them:

1. Edit `.env` file in the root directory
2. Update `ROBOFLOW_API_KEY` and `ROBOFLOW_MODEL_URL`
3. Restart the backend server

## Features
- 🔍 AI-powered mold detection
- 📊 Interactive analytics dashboard
- 📝 Analysis history tracking
- 🎨 Modern responsive UI
- 💡 Smart recommendations
- 🔄 Label toggle functionality

## Troubleshooting
- If PowerShell blocks npm, use Command Prompt instead
- Ensure both backend (port 8000) and frontend (port 3000) are running
- Check that the `.env` file exists in the root directory