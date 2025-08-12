import os
import cv2
import numpy as np
import pytesseract
from sympy import symbols, Eq, solve, sympify, SympifyError

def preprocess_image(image_path):
    """
    Preprocess the image for better OCR results:
    1. Convert to grayscale
    2. Resize (2x upscaling)
    3. Apply Gaussian blur to remove noise
    4. Apply adaptive thresholding
    
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
    
    # Resize (2x upscaling)
    height, width = gray.shape
    gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to get a binary image
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Alternative: Otsu's thresholding
    # _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
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
    # OCR configuration for math equations
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789+-*/()=xX^." '
    
    try:
        # Perform OCR
        ocr_data = pytesseract.image_to_data(cv2.imread(image_path), config=custom_config, output_type=pytesseract.Output.DICT)
    except pytesseract.pytesseract.TesseractNotFoundError:
        raise Exception(r"C:\Program Files\Tesseract-OCR\tesseract.exe is not installed or it's not in your PATH. See README file for more information.")
    
    # Extract text and confidence
    texts = [word for word in ocr_data['text'] if word.strip()]
    confidences = [conf for word, conf in zip(ocr_data['text'], ocr_data['conf']) if word.strip()]
    
    if not texts:
        return "", 0
    
    # Join all detected text parts
    equation_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Clean and format the extracted text
    equation_text = clean_equation_text(equation_text)
    
    return equation_text, avg_confidence

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
    # Remove extra spaces
    text = " ".join(text.split())
    
    # Handle common OCR mistakes
    replacements = {
        'O': '0',  # Replace O with 0
        'o': '0',  # Replace o with 0
        'l': '1',  # Replace l with 1
        'I': '1',  # Replace I with 1
        ',': '.',  # Replace comma with decimal point
        '÷': '/',  # Replace division symbol with slash
        '×': '*',  # Replace multiplication symbol with asterisk
        '^': '**',  # Replace caret with double asterisk for power
        '.': '.',  # Keep decimal points
        '"': '',   # Remove quotes
        "'": '',   # Remove single quotes
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
    text = re.sub(r'([a-zA-Z])\s+(\d)', r'\1*\2', text)
    
    # Add multiplication between number and variable (e.g., "2 x" becomes "2*x")
    text = re.sub(r'(\d)\s+([a-zA-Z])', r'\1*\2', text)
    
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