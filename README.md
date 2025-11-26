# Image Colorization - ML Project

## Project Structure

```
├── training/                      # Model training code
│   ├── model.py                   # Jupyter notebook with model architecture + training code. Saves model as .pth file
│
└── web_app/                       # Web application
    ├── backend/                   # Flask API
    │   ├── app.py                 # API endpoints for image upload + model inference
    │   ├── model.py               # Utility functions for loading model from downloaded .pth files + image processing
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
- Save trained model as `.pth` file (to be used by backend)

**Backend** (`/web_app/backend`):
- Flask API that loads model from `.pth` files on startup
- `/colourize` endpoint: accepts image → returns colorized image
- Model loaded once on startup, persists in memory
- To run:
  ```bash
  cd web_app/backend
  pip install -r requirements.txt
  python app.py
  ```

**Frontend** (`/web_app/frontend`):
- Web UI for uploading grayscale images
- Calls backend `/colourize` endpoint
- Displays colorized results

## Key Design Notes

- **Single Git Repo**: Training code and web app are in the same repository
- **Model Storage**: Model is trained using google colab, saved as `.pth` file, and loaded by backend
- **No Retraining**: Backend loads model once on startup and keeps it in memory
- **Scalable**: Model persists across requests, no redundant loading
