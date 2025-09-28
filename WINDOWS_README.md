# Handwritten Equation Solver - Windows Setup Guide

A Flask web application that recognizes and solves handwritten mathematical equations from images using OpenCV, Tesseract OCR, and SymPy.

## Features
- ✅ Upload images of handwritten equations
- ✅ Draw equations directly on a canvas
- ✅ Image preprocessing for better OCR accuracy
- ✅ Equation recognition using Tesseract OCR
- ✅ Solve algebraic equations and evaluate expressions
- ✅ Responsive UI with Bootstrap
- ✅ Drag-and-drop image upload

## Quick Start (Windows)

### Prerequisites
- Python 3.7+ installed
- Git (for cloning)
- Tesseract OCR ✅ (already installed on this system)

### Running the Application

1. **Double-click `run_app.bat`** to start the application automatically
   
   OR

2. **Manual start:**
   ```cmd
   # Navigate to the project directory
   cd "C:\Users\drsan\hand"
   
   # Activate virtual environment
   venv\Scripts\activate
   
   # Run the application
   python app.py
   ```

3. **Open your browser** and go to: http://127.0.0.1:5000

4. **Stop the application** by pressing `Ctrl+C` in the terminal

### Usage

1. **Upload Image**: Click "Upload Image" tab and select/drag an image file
2. **Draw Equation**: Click "Draw Equation" tab and draw directly on the canvas
3. **View Results**: See the recognized equation and its solution

### Supported File Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

### Example Equations
- Simple algebra: `2x + 3 = 7` → `x = 2`
- Basic arithmetic: `5 + 3 * 2` → `11`
- Quadratic: `x^2 - 4 = 0` → `x = -2, 2`

### Troubleshooting

**Application won't start:**
- Ensure you're in the correct directory
- Check that Python is installed: `python --version`
- Try running `venv\Scripts\activate` manually first

**Poor OCR recognition:**
- Use clear, well-written equations
- Ensure good lighting and contrast
- Avoid complex mathematical notation
- Use simple variable names (x, y)

**Tesseract errors:**
- The application is configured for the default Tesseract installation
- Current path: `C:\Program Files\Tesseract-OCR\tesseract.exe` ✅

### Technical Details

**Dependencies:**
- Flask (web framework)
- OpenCV (image processing)
- Tesseract + pytesseract (OCR)
- SymPy (equation solving)
- NumPy, Pillow (image handling)

**Project Structure:**
```
hand/
├── app.py                 # Flask application
├── utils.py              # Image processing & solver functions
├── requirements.txt      # Python dependencies
├── run_app.bat           # Windows startup script
├── venv/                 # Virtual environment
├── static/
│   ├── uploads/          # Uploaded images
│   └── styles.css        # Custom styling
└── templates/
    ├── index.html        # Main page
    └── result.html       # Results page
```

### Limitations
- Works best with clear, well-written equations
- Limited to algebraic equations with one variable (x)
- May struggle with complex mathematical notation
- OCR accuracy depends on image quality

---

*Application successfully set up and tested on Windows 11 with Python 3.13 and Tesseract OCR 5.3.3*
