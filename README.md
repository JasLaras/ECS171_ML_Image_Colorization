# Image Colorization - ML Project

## Project Structure

```
├── training/                      # Model training code
│   ├── model.py                   # Model architecture (link to Colab notebook or training workflow)
│
└── web_app/                       # Web application
    ├── backend/                   # Flask API
    │   ├── app.py                 # API endpoints for image upload + model inference
    │   └── requirements.txt       # Python dependencies for backend
    │
    └── frontend/                  # Web UI
        ├── project.html           # Landing page with project selection bubbles
        ├── image-upload.html      # Image recolorization upload page (grayscale → RGB)
        ├── model.html             # Page with notes on model design and architecture explanation
        ├── report.html            # Page for final research report
        ├── style.css              # Shared styling 
        ├── image-upload.css       # Styling specific to image-upload page (upload box behavior, layout)
        ├── upload.js              # Handles file preview + sending images to backend
        ├── script.js              # General UI interaction logic for project.html 
```

## Architecture

**Training** (`/training`):
- Develop and train the colorization model
- Save trained model to Hugging Face Hub

**Backend** (`/web_app/backend`):
- Flask API that loads model from Hugging Face
- `/predict` endpoint: accepts image → returns colorized image
- Model loaded once on startup, persists in memory

**Frontend** (`/web_app/frontend`):
- Web UI for uploading grayscale images
- Calls backend `/predict` endpoint
- Displays colorized results

## Key Design Notes

- **Single Git Repo**: Training code and web app are in the same repository
- **Model Storage**: Trained model is uploaded to Hugging Face Model Hub
- **No Retraining**: Backend downloads model once and keeps it in memory
- **Scalable**: Model persists across requests, no redundant loading
