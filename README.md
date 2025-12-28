# Bread Mold Detection System

A modern, professional AI-powered bread mold detection system with FastAPI backend and Node.js frontend.

## Features

-  **AI-Powered Detection**: Uses Roboflow model for accurate mold detection
-  **Analytics Dashboard**: Interactive charts with pastel blue/orange theme
-  **History Tracking**: View all past analyses with detailed breakdowns
-  **Modern UI**: Lively gradient backgrounds with Space Grotesk font
-  **Detailed Analysis**: Bread type, mold type, storage time, and age estimation
-  **Smart Recommendations**: Storage tips and mold information
-  **Label Toggle**: Show/hide detection labels on images

## Tech Stack

### Backend (FastAPI)
- FastAPI for high-performance API
- Roboflow integration for ML model
- PIL for image processing
- JSON-based history storage

### Frontend (Node.js + Vanilla JS)
- Express.js server
- Chart.js for interactive visualizations
- Modern vanilla JavaScript
- Space Grotesk font
- Pastel blue & orange color theme

## Usage

1. Start the backend server (port 8000)
2. Start the frontend server (port 3000)
3. Open http://localhost:3000 in your browser
4. Upload a bread image
5. Click "Analyze Image"
6. View detailed results including:
   - Risk level and mold coverage (with modern gradient)
   - Bread and mold types (underscores removed)
   - Estimated storage time and age in days
   - Recommended actions (elaborated)
   - Storage recommendations
   - Mold information
7. Toggle labels on/off on the annotated image
8. View analytics with interactive charts
9. Check history of all analyses

## API Endpoints

- `POST /api/analyze` - Analyze bread image
- `GET /api/history` - Get analysis history
- `GET /api/analytics` - Get analytics data with category breakdowns

## Model Logic

The system calculates:
- **Mold Coverage**: Pixel-based calculation with overlap handling
- **Risk Level**: None, Low, Moderate, Severe based on coverage
- **Storage Time**: Estimated from mold growth patterns
- **Bread Age**: Days estimation (1, 4, 6, 10) + classification
- **Verdict**: Healthy or Not Healthy with detailed actions
- **Recommendations**: Storage tips and mold information
