<<<<<<< HEAD
# 🍞 How to Run the Bread Mold Detection System

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- A Roboflow account with API access

---

## 📋 Step-by-Step Setup

### 1. Clone or Download the Repository

```bash
cd Bread-Mold-CS
```

### 2. Set Up Environment Variables

**IMPORTANT:** You need to configure your Roboflow API credentials.

1. Copy the example environment file:
```bash
copy .env.example .env
```

2. Open `.env` file in a text editor and add your credentials:
```
ROBOFLOW_API_KEY=your_actual_api_key_here
ROBOFLOW_MODEL_URL=your_actual_model_url_here
```

**Where to get these:**
- Log in to your Roboflow account
- Go to your project settings
- Copy your API key and model URL

### 3. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Required packages:**
- fastapi
- uvicorn
- python-multipart
- pillow
- requests
- python-dotenv

### 4. Start the Backend Server

```bash
python main.py
```

**Expected output:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The backend API will be available at: **http://localhost:8000**

---

## 🚀 Running the Application

### Option 1: Using the FastAPI Backend (Recommended)

1. **Start the backend** (if not already running):
```bash
cd backend
python main.py
```

2. **Open the web interface**:
```bash
cd bread_mold_webapp
python -m http.server 8080
```

3. **Access the application**:
   - Open your browser and go to: **http://localhost:8080**
   - Or open `bread_mold_webapp/templates/index.html` directly in your browser

### Option 2: Direct File Access

Simply open `bread_mold_webapp/templates/index.html` in your web browser.

---

## 🧪 Testing the Application

1. **Upload an image:**
   - Click "Choose File" button
   - Select a bread image from your computer
   - Click "Analyze" button

2. **View results:**
   - Detection Summary (Risk, Coverage, Bread Type, Mold Type)
   - Annotated Image with detection boxes
   - Final Verdict (Healthy/Not Healthy)
   - Result Breakdown (4 detailed cards)
   - Mold Information (if mold detected)
   - Recommended Actions

---

## 📊 API Endpoints

### Analyze Image
```
POST http://localhost:8000/api/analyze
Content-Type: multipart/form-data
Body: image file
```

### Get History
```
GET http://localhost:8000/api/history
```

### Get Analytics
```
GET http://localhost:8000/api/analytics
```

---

## 🔧 Troubleshooting

### Backend won't start
- **Error:** `ROBOFLOW_API_KEY and ROBOFLOW_MODEL_URL must be set`
  - **Solution:** Make sure `.env` file exists and contains valid credentials

### CORS errors in browser
- **Error:** `Access to fetch blocked by CORS policy`
  - **Solution:** Make sure backend is running on port 8000

### Module not found errors
- **Error:** `ModuleNotFoundError: No module named 'fastapi'`
  - **Solution:** Run `pip install -r requirements.txt` in the backend folder

### Image upload fails
- **Error:** `Failed to analyze image`
  - **Solution:** Check that your Roboflow API key is valid and has sufficient credits

---

## 🎨 Features Overview

### Detection Summary
- Horizontal card layout showing key metrics at a glance
- Risk Level, Mold Coverage, Bread Type, Mold Type

### Main Content
- Side-by-side image and verdict display
- Annotated image with detection boxes
- Final verdict with storage time and bread age

### Result Breakdown
- 4 detailed cards with comprehensive information
- Detection, Bread Info, Mold Info, Status

### Mold Information
- Dynamic content based on detected mold type
- Health risks and characteristics
- Only shown when mold is detected

### Recommended Actions
- Immediate action steps
- Storage tips based on bread type

---

## 📁 Project Structure

```
Bread-Mold-CS/
├── backend/
│   ├── main.py              # FastAPI backend
│   ├── requirements.txt     # Python dependencies
│   └── analysis_history.json # Analysis history storage
├── bread_mold_webapp/
│   ├── templates/
│   │   └── index.html       # Main web interface
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css    # Styling
│   │   └── js/
│   │       └── app.js       # Frontend logic
├── .env.example             # Environment template
├── .env                     # Your credentials (DO NOT COMMIT)
└── HOW_TO_RUN.md           # This file
```

---

## 🔒 Security Notes

- ✅ Never commit `.env` file to version control
- ✅ Keep your API keys private
- ✅ Use `.env.example` as a template only
- ✅ The `.env` file is already in `.gitignore`

---

## 💡 Tips

1. **Best image quality:** Use clear, well-lit photos of bread
2. **Optimal size:** Images are automatically resized to 800px max
3. **Multiple analyses:** History is saved automatically (last 100 scans)
4. **Storage recommendations:** Vary by bread type detected

---

## 📞 Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify your `.env` file has correct credentials
3. Ensure backend is running on port 8000
4. Check browser console for error messages

---

## 🎉 You're Ready!

Your Bread Mold Detection System is now set up and ready to use. Happy analyzing! 🍞🔍
=======
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
>>>>>>> 33f06c0bc8d6ee6dbcc4b1025b3394ca04dbbfd4
