import os
import cv2
import numpy as np
import pytesseract
from sympy import symbols, Eq, solve, sympify, SympifyError

def preprocess_image(image_path):
    """
    Preprocess the image for better OCR results:
    1. Convert to grayscale
    2. Resize for optimal OCR
    3. Apply denoising
    4. Apply thresholding
    5. Morphological operations to clean up
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Path to the preprocessed image
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize for better OCR (scale up small images, ensure minimum size)
    height, width = gray.shape
    min_height, min_width = 200, 400
    
    if height < min_height or width < min_width:
        # Scale up small images
        scale_factor = max(min_height / height, min_width / width, 2.0)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply denoising
    denoised = cv2.medianBlur(gray, 3)
    
    # Try multiple thresholding approaches and pick the best
    # 1. Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # 2. Otsu's thresholding
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use adaptive for most cases, as it handles varying lighting better
    binary = adaptive
    
    # Morphological operations to clean up noise
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Save the preprocessed image
    preprocessed_path = os.path.splitext(image_path)[0] + '_preprocessed.png'
    cv2.imwrite(preprocessed_path, binary)
    
    return preprocessed_path

def extract_equation(image_path):
    """
    Extract equation text from the preprocessed image using OCR
    
    Args:
        image_path: Path to the preprocessed image
        
    Returns:
        tuple: (equation_text, confidence)
    """
    # Check if image file exists and is readable
    if not os.path.exists(image_path):
        return "", 0
    
    try:
        # Load image for OCR
        image = cv2.imread(image_path)
        if image is None:
            return "", 0
        
        # Try multiple OCR configurations
        configs = [
            '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789+-*/()=xX^.-',  # Single word
            '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789+-*/()=xX^.-',  # Single text line
            '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/()=xX^.-',  # Uniform block
            '--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789+-*/()=xX^.-', # Raw line
            '--oem 3 --psm 8',  # Single word, no whitelist
            '--oem 3 --psm 7',  # Single text line, no whitelist
        ]
        
        best_text = ""
        best_confidence = 0
        
        for i, config in enumerate(configs):
            try:
                # Try simple OCR first
                simple_text = pytesseract.image_to_string(image, config=config).strip()
                
                if simple_text and len(simple_text) > len(best_text):
                    best_text = simple_text
                    
                # Try detailed OCR
                ocr_data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
                
                # Extract non-empty text with confidence
                texts = []
                confidences = []
                for j, word in enumerate(ocr_data['text']):
                    if word.strip() and ocr_data['conf'][j] > 0:
                        texts.append(word.strip())
                        confidences.append(ocr_data['conf'][j])
                
                if texts:
                    combined_text = " ".join(texts)
                    avg_conf = sum(confidences) / len(confidences)
                    
                    if avg_conf > best_confidence or (avg_conf >= best_confidence and len(combined_text) > len(best_text)):
                        best_text = combined_text
                        best_confidence = avg_conf
                        
            except Exception:
                continue
        
        if not best_text:
            return "", 0
        
        # Clean and format the extracted text
        equation_text = clean_equation_text(best_text)
        
        return equation_text, best_confidence
        
    except pytesseract.pytesseract.TesseractNotFoundError:
        raise Exception(r"C:\Program Files\Tesseract-OCR\tesseract.exe is not installed or it's not in your PATH. See README file for more information.")
    except Exception:
        return "", 0

def clean_equation_text(text):
    """
    Clean and format the extracted text:
    1. Remove extra spaces
    2. Handle common OCR mistakes
    3. Fix spacing issues between numbers and variables
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned equation text
    """
    if not text:
        return ""
    
    # Remove extra spaces
    text = " ".join(text.split())
    
    # Handle common OCR mistakes
    replacements = {
        'O': '0',   # Replace O with 0
        'o': '0',   # Replace o with 0
        'l': '1',   # Replace l with 1
        'I': '1',   # Replace I with 1
        'S': '5',   # Replace S with 5 (common mistake)
        'Z': '2',   # Replace Z with 2 (sometimes)
        'g': '9',   # Replace g with 9
        ',': '.',   # Replace comma with decimal point
        '÷': '/',   # Replace division symbol with slash
        '×': '*',   # Replace multiplication symbol with asterisk
        '·': '*',   # Replace middle dot with asterisk
        '^': '**',  # Replace caret with double asterisk for power
        '"': '',    # Remove quotes
        "'": '',    # Remove single quotes
        '`': '',    # Remove backticks
        '_': '',    # Remove underscores (often OCR noise)
        '|': '1',   # Replace pipe with 1
        'T': '7',   # Replace T with 7 (sometimes)
        '—': '-',   # Replace em-dash with minus
        '–': '-',   # Replace en-dash with minus
        '−': '-',   # Replace Unicode minus with regular minus
        '‐': '-',   # Replace hyphen with minus
        '‑': '-',   # Replace non-breaking hyphen with minus
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove spaces around operators
    operators = ['+', '-', '*', '/', '=', '(', ')']
    for op in operators:
        text = text.replace(f" {op} ", op)
        text = text.replace(f" {op}", op)
        text = text.replace(f"{op} ", op)
    
    # Handle spaces between numbers and variables (e.g., "x 11" should be "x*11")
    import re
    
    # Add multiplication between variable and number (e.g., "x 2" becomes "x*2")
    text = re.sub(r'([a-zA-Z])\s*(\d)', r'\1*\2', text)
    
    # Add multiplication between number and variable (e.g., "2 x" becomes "2*x")
    text = re.sub(r'(\d)\s*([a-zA-Z])', r'\1*\2', text)
    
    # Handle cases like "2x" -> "2*x" (no space between number and variable)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    
    # Remove any remaining spaces
    text = text.replace(" ", "")
    
    # Ensure 'x' is lowercase for sympy
    text = text.replace('X', 'x')
    
    # Remove trailing periods that might be added by OCR
    if text.endswith('.'):
        text = text[:-1]
    
    return text

def solve_equation(equation_text):
    """
    Parse and solve the equation using sympy
    
    Args:
        equation_text: Cleaned equation text
        
    Returns:
        tuple: (result, error_message)
    """
    if not equation_text or equation_text.isspace():
        return None, "No equation detected"
    
    try:
        # Define symbol
        x = symbols('x')
        
        # Check if it's an equation (contains '=')
        if '=' in equation_text:
            # Split into left and right sides
            left_side, right_side = equation_text.split('=')
            
            try:
                # Parse into sympy equation
                left_expr = sympify(left_side.strip())
                right_expr = sympify(right_side.strip())
                equation = Eq(left_expr, right_expr)
                
                # Solve for x
                solution = solve(equation, x)
                
                if solution:
                    return solution, None
                else:
                    return None, "Could not find a solution"
            except SympifyError as e:
                return None, f"Error parsing equation: {str(e)}"
        else:
            # Evaluate as a mathematical expression
            try:
                result = sympify(equation_text)
                return result, None
            except SympifyError as e:
                return None, f"Error evaluating expression: {str(e)}"
    except Exception as e:
        return None, f"Error solving equation: {str(e)}"
