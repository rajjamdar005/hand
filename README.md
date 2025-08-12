# Handwritten Equation Solver

A Flask web application that recognizes and solves handwritten mathematical equations from images. The application uses OpenCV for image preprocessing, Tesseract OCR for text recognition, and SymPy for equation solving.

## Features

- Upload images of handwritten equations
- Draw equations directly on a canvas
- Image preprocessing for better OCR accuracy
- Equation recognition using Tesseract OCR
- Solving algebraic equations and evaluating expressions using SymPy
- Responsive UI with Bootstrap
- Drag-and-drop image upload

## Project Structure

```
handwritten_solver/
├── app.py               # Flask main app
├── requirements.txt     # Dependencies
├── utils.py             # Image processing & solver functions
├── static/
│   ├── uploads/         # Uploaded images
│   └── styles.css       # Custom styling
├── templates/
│   ├── index.html       # Upload form
│   └── result.html      # Result display
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/rajjamdar005/hand
cd hand
```

2. Create a virtual environment and activate it:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:

### Windows

1. Download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer and follow the instructions
3. **IMPORTANT**: Make sure to install Tesseract to the default location `C:\Program Files\Tesseract-OCR\`
4. Add Tesseract to your PATH:
   - Right-click on 'This PC' or 'My Computer' and select 'Properties'
   - Click on 'Advanced system settings'
   - Click on 'Environment Variables'
   - Under 'System variables', find and select 'Path', then click 'Edit'
   - Click 'New' and add `C:\Program Files\Tesseract-OCR\`
   - Click 'OK' on all dialogs to save the changes

**Note**: The application is configured to look for Tesseract at `C:\Program Files\Tesseract-OCR\tesseract.exe`. If you installed it to a different location, you'll need to update the path in `app.py`

### macOS

```bash
brew install tesseract
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install tesseract-ocr
```

## Usage

1. Start the Flask application:

```bash
python app.py
```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`

3. Upload an image of a handwritten equation or draw one directly on the canvas

4. View the recognized equation and its solution

## Example

1. Upload or draw a simple equation like `2x + 3 = 7`
2. The application will preprocess the image, recognize the equation, and solve it to get `x = 2`

## Limitations

- Works best with clear, well-written equations
- May struggle with complex mathematical notation or symbols
- Limited to algebraic equations with one variable (x) and basic arithmetic expressions

## Troubleshooting

### Tesseract OCR Installation Issues

If you encounter an error like `C:\Program Files\Tesseract-OCR\tesseract.exe is not installed or it's not in your PATH`, follow these steps:

1. Verify that Tesseract OCR is installed correctly:
   - Check if the directory `C:\Program Files\Tesseract-OCR\` exists
   - Confirm that `tesseract.exe` is present in that directory

2. If Tesseract is installed in a different location, update the path in `app.py`:
   ```python
   # Find this line in app.py
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   # Change it to your actual installation path
   pytesseract.pytesseract.tesseract_cmd = r"Your\Actual\Path\To\tesseract.exe"
   ```

3. Ensure Tesseract is in your system PATH (as described in the installation section)

4. Restart your application after making any changes

## Future Improvements

- Support for more complex mathematical notation
- Multiple variable support
- Improved OCR accuracy for mathematical symbols
- Step-by-step solution display

## License

MIT