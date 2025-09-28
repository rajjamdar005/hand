from flask import Flask, render_template, request, redirect, url_for, flash
import google.generativeai as genai
import os
import uuid
import logging

logging.basicConfig(level=logging.DEBUG, filename='d:\hand\ocr_debug.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("App.py: Logging configured and working.")

# Explicitly set the logger level for equation_solver
logging.getLogger('equation_solver').setLevel(logging.DEBUG)

from utils import preprocess_image, extract_equation, solve_equation

app = Flask(__name__)
app.secret_key = 'handwritten_equation_solver'

# Configure upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'equation_image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['equation_image']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and file.filename:
        # Check if file is an image
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            flash('Please upload a valid image file (jpg, png, gif, bmp, tiff, webp)')
            return redirect(request.url)
        
        # Generate unique filename to prevent overwriting
        filename = str(uuid.uuid4()) + file_ext
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Preprocess the image (now returns multiple versions)
            preprocessed_paths = preprocess_image(file_path)
            
            # Extract equation text using OCR (tries all preprocessed versions)
            equation_text, confidence = extract_equation(preprocessed_paths)
            
            # Solve the equation
            result, error = solve_equation(equation_text, model)
            
            # Handle errors gracefully - don't show technical errors to the user
            if error and "could not parse" in error:
                error = "Unable to process this equation. Please try writing it more clearly."
            elif error and "SyntaxError" in error:
                error = "Unable to understand the equation format. Please check your handwriting."
            elif error and "TokenError" in error:
                error = "Unable to process this equation. Please try again."
            
            # Use the first preprocessed image for display
            display_preprocessed = preprocessed_paths[0] if preprocessed_paths else file_path
            
            return render_template('result.html', 
                                   original_image=file_path.replace('\\', '/'),
                                   preprocessed_image=display_preprocessed.replace('\\', '/'),
                                   equation=equation_text,
                                   result=result,
                                   error=error,
                                   confidence=confidence)
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))

@app.route('/canvas_upload', methods=['POST'])
def canvas_upload():
    # Get the base64 image data from the request
    image_data = request.form.get('image_data')
    if not image_data:
        flash('No image data received')
        return redirect(url_for('index'))
    
    try:
        # Remove the data URL prefix (e.g., 'data:image/png;base64,')
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        import base64
        from io import BytesIO
        from PIL import Image
        
        # Decode the base64 image
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        
        # Save the image
        filename = str(uuid.uuid4()) + '.png'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(file_path)
        
        # Process the image as with file upload
        preprocessed_paths = preprocess_image(file_path)
        equation_text, confidence = extract_equation(preprocessed_paths)
        result, error = solve_equation(equation_text, model)
        
        # Handle errors gracefully - don't show technical errors to the user
        if error and "could not parse" in error:
            error = "Unable to process this equation. Please try writing it more clearly."
        elif error and "SyntaxError" in error:
            error = "Unable to understand the equation format. Please check your handwriting."
        elif error and "TokenError" in error:
            error = "Unable to process this equation. Please try again."
        
        # Use the first preprocessed image for display
        display_preprocessed = preprocessed_paths[0] if preprocessed_paths else file_path
        
        return render_template('result.html', 
                               original_image=file_path.replace('\\', '/'),
                               preprocessed_image=display_preprocessed.replace('\\', '/'),
                               equation=equation_text,
                               result=result,
                               error=error,
                               confidence=confidence)
    except Exception as e:
        flash(f'Error processing canvas image: {str(e)}')
        return redirect(url_for('index'))

# For Windows, set the Tesseract path
import platform
if platform.system() == 'Windows':
    import pytesseract
    import os
    import subprocess
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    # Check if Tesseract is installed at the expected path
    if not os.path.exists(tesseract_path):
        print("\n" + "*" * 80)
        print("WARNING: Tesseract OCR not found at the expected location!")
        print(f"Expected path: {tesseract_path}")
        
        # Check if the installer script exists
        installer_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "install_tesseract.py")
        if os.path.exists(installer_script):
            print("\nAn automatic installer for Tesseract OCR is available.")
            response = input("Would you like to run the installer now? (y/n): ")
            if response.lower() == 'y':
                print("\nLaunching installer with administrator privileges...")
                print("Please approve the UAC prompt if it appears.")
                try:
                    # Run the installer script with admin privileges
                    subprocess.run(["powershell", "Start-Process", "python", "-ArgumentList", f"\"{installer_script}\"", "-Verb", "RunAs"])
                    print("\nAfter installation completes, please restart this application.")
                    input("Press Enter to exit...")
                    exit(0)
                except Exception as e:
                    print(f"Error launching installer: {e}")
        
        print("\nAlternatively, you can manually install Tesseract OCR:")
        print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Install to the default location")
        print("3. Add to PATH as described in the README.md file")
        print("*" * 80 + "\n")
    
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Configure Gemini AI model
genai.configure(api_key="AIzaSyDAv1fbNfJ4na_ns5JSZ01Rgrv0VNGeLw0")
model = genai.GenerativeModel('gemini-2.5-flash')

if __name__ == '__main__':
    app.run(debug=True)