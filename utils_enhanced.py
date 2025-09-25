import os
import cv2
import numpy as np
import pytesseract
from sympy import symbols, Eq, solve, sympify, SympifyError
import re

def preprocess_image(image_path):
    """
    Enhanced preprocessing for handwritten equations with multiple approaches
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize for better OCR (scale up small images)
    height, width = gray.shape
    min_height, min_width = 300, 600
    
    if height < min_height or width < min_width:
        scale_factor = max(min_height / height, min_width / width, 3.0)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Create multiple preprocessed versions
    preprocessed_images = []
    
    # Version 1: Standard approach for black text on white background
    if np.mean(gray) > 127:  # Light background
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # Adaptive threshold for black text on white
        binary1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        preprocessed_images.append(("standard", binary1))
    
    # Version 2: Inverted approach for white text on dark background
    if np.mean(gray) < 127:  # Dark background
        # Invert the image first
        inverted = cv2.bitwise_not(gray)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(inverted, (3, 3), 0)
        # Adaptive threshold
        binary2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        preprocessed_images.append(("inverted", binary2))
    
    # Version 3: Otsu's thresholding
    # Apply slight blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary3 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(("otsu", binary3))
    
    # Version 4: For very thick handwritten text
    # Morphological operations to separate touching characters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # For light backgrounds
    if np.mean(gray) > 127:
        binary4 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    else:
        # For dark backgrounds
        inverted = cv2.bitwise_not(gray)
        binary4 = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    
    # Apply morphological opening to separate characters
    binary4 = cv2.morphologyEx(binary4, cv2.MORPH_OPEN, kernel, iterations=1)
    # Apply closing to fill gaps
    binary4 = cv2.morphologyEx(binary4, cv2.MORPH_CLOSE, kernel, iterations=1)
    preprocessed_images.append(("morphological", binary4))
    
    # Save all versions and return the best one
    base_path = os.path.splitext(image_path)[0]
    saved_paths = []
    
    for name, img in preprocessed_images:
        path = f"{base_path}_preprocessed_{name}.png"
        cv2.imwrite(path, img)
        saved_paths.append(path)
    
    # Return the first one as default, but we'll try all in OCR
    return saved_paths

def extract_equation(image_paths):
    """
    Enhanced OCR with multiple preprocessing approaches and validation
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    best_result = ""
    best_confidence = 0
    
    # OCR configurations optimized for equations
    configs = [
        # For isolated characters/words
        '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789+-*/()=xX^.', 
        # For single text lines  
        '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789+-*/()=xX^.',
        # For uniform blocks
        '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/()=xX^.',
        # Raw line, treat as single text line
        '--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789+-*/()=xX^.',
        # Single character mode (for very simple equations)
        '--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789+-*/()=xX^.',
        # No whitelist versions (sometimes whitelist is too restrictive)
        '--oem 3 --psm 8',
        '--oem 3 --psm 7',
        '--oem 3 --psm 6',
    ]
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
            
        try:
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            for config in configs:
                try:
                    # Try OCR with this config
                    text = pytesseract.image_to_string(image, config=config).strip()
                    
                    if not text:
                        continue
                    
                    # Get confidence
                    try:
                        ocr_data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
                        confidences = [conf for conf in ocr_data['conf'] if conf > 0]
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    except:
                        avg_confidence = 50  # Default confidence
                    
                    # Validate if this looks like an equation
                    if is_valid_equation(text) and avg_confidence > best_confidence:
                        best_result = text
                        best_confidence = avg_confidence
                    
                    # Also consider longer results with decent confidence
                    if len(text) > len(best_result) and avg_confidence > 30:
                        best_result = text
                        best_confidence = avg_confidence
                        
                except Exception:
                    continue
        except Exception:
            continue
    
    if not best_result:
        return "", 0
    
    # Clean and return the best result
    cleaned_equation = clean_equation_text(best_result)
    return cleaned_equation, best_confidence

def is_valid_equation(text):
    """
    Validate if OCR result looks like a mathematical equation
    """
    if not text or len(text.strip()) < 1:
        return False
    
    # Check for mathematical characters
    math_chars = set('0123456789+-*/()=xX^.')
    text_chars = set(text.replace(' ', ''))
    
    # At least 80% of characters should be mathematical
    if len(text_chars) > 0:
        math_ratio = len(text_chars & math_chars) / len(text_chars)
        if math_ratio < 0.6:
            return False
    
    # Should not be just numbers (like "442")
    if text.replace(' ', '').isdigit() and len(text.replace(' ', '')) > 2:
        return False
    
    # Should contain at least one number or variable
    if not any(c.isdigit() or c.lower() == 'x' for c in text):
        return False
    
    return True

def clean_equation_text(text):
    """
    Enhanced cleaning for handwritten text
    """
    if not text:
        return ""
    
    # Remove extra spaces
    text = " ".join(text.split())
    
    # Enhanced OCR mistake corrections
    replacements = {
        'O': '0', 'o': '0', 'Q': '0',  # O variations to 0
        'l': '1', 'I': '1', '|': '1', 'i': '1',  # l variations to 1
        'S': '5', 's': '5',  # S to 5
        'Z': '2', 'z': '2',  # Z to 2
        'g': '9', 'q': '9',  # g to 9
        'G': '6',  # G to 6
        'B': '8',  # B to 8
        'T': '7', 't': '7',  # T to 7
        ',': '.', ';': '.',  # Comma/semicolon to decimal
        '÷': '/', '×': '*', '·': '*',  # Math symbols
        '^': '**',  # Exponent
        '"': '', "'": '', '`': '', '_': '',  # Remove noise
        '—': '-', '–': '-', '−': '-', '‐': '-', '‑': '-',  # Dashes to minus
    }
    
    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove spaces around operators
    operators = ['+', '-', '*', '/', '=', '(', ')']
    for op in operators:
        text = text.replace(f" {op} ", op)
        text = text.replace(f" {op}", op)
        text = text.replace(f"{op} ", op)
    
    # Handle implicit multiplication
    # Number + letter (e.g., "2x" -> "2*x")
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    # Letter + number (e.g., "x2" -> "x*2")  
    text = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', text)
    
    # Remove any remaining spaces
    text = text.replace(" ", "")
    
    # Ensure lowercase x for variables
    text = text.replace('X', 'x')
    
    # Remove trailing periods
    if text.endswith('.'):
        text = text[:-1]
    
    # Special case: if result is just numbers without operators, try to parse as addition
    if re.match(r'^\d{2,}$', text) and len(text) <= 4:
        # For cases like "442" from "1+2", try to interpret
        if len(text) == 3:
            # Could be "1+2" misread as "442" or "142" etc.
            # This is tricky - we'll return as-is and let validation catch it
            pass
    
    return text

def solve_equation(equation_text):
    """
    Enhanced equation solver with better error handling
    """
    if not equation_text or equation_text.isspace():
        return None, "No equation detected"
    
    # Pre-validate the equation
    if not is_valid_equation(equation_text):
        return None, f"Invalid equation format: '{equation_text}'"
    
    try:
        x = symbols('x')
        
        if '=' in equation_text:
            # Handle equations
            parts = equation_text.split('=')
            if len(parts) != 2:
                return None, "Invalid equation format"
            
            left_side, right_side = parts[0].strip(), parts[1].strip()
            
            try:
                left_expr = sympify(left_side)
                right_expr = sympify(right_side)
                equation = Eq(left_expr, right_expr)
                
                solution = solve(equation, x)
                if solution:
                    return solution, None
                else:
                    return None, "Could not find a solution"
            except SympifyError as e:
                return None, f"Error parsing equation: {str(e)}"
        else:
            # Handle expressions
            try:
                result = sympify(equation_text)
                return result, None
            except SympifyError as e:
                return None, f"Error evaluating expression: {str(e)}"
    except Exception as e:
        return None, f"Error solving equation: {str(e)}"
