import os
import cv2
import numpy as np
import pytesseract
from sympy import symbols, Eq, solve, sympify, SympifyError
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)


def preprocess_image(image_path):
    """
    Enhanced preprocessing for better OCR results:
    1. Convert to grayscale
    2. Resize for optimal OCR (scale up small images, ensure minimum size)
    3. Apply denoising
    4. Apply thresholding
    5. Morphological operations to clean up
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Path to the preprocessed image
    """
    print(f"[DEBUG] Preprocessing image: {image_path}")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    print(f"[DEBUG] Original image shape: {image.shape}")
    
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
    
    print(f"[DEBUG] Resized image shape: {gray.shape}")
    
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
    
    print(f"[DEBUG] Preprocessed image saved: {preprocessed_path}")
    return preprocessed_path

def extract_equation(image_path):
    """
    Extract equation text from the preprocessed image using OCR
    
    Args:
        image_path: Path to the preprocessed image
        
    Returns:
        tuple: (equation_text, confidence)
    """
    print(f"[DEBUG] Processing image: {image_path}")
    
    # Check if image file exists and is readable
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        return "", 0
    
    try:
        # Load image for OCR
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return "", 0
            
        print(f"[DEBUG] Image loaded successfully. Shape: {image.shape}")
        
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
                print(f"[DEBUG] Config {i+1} simple result: '{simple_text}'")
                
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
                    print(f"[DEBUG] Config {i+1} detailed result: '{combined_text}' (confidence: {avg_conf:.1f})")
                    
                    if avg_conf > best_confidence or (avg_conf >= best_confidence and len(combined_text) > len(best_text)):
                        best_text = combined_text
                        best_confidence = avg_conf
                        
            except Exception as e:
                print(f"[DEBUG] Config {i+1} failed: {e}")
                continue
        
        if not best_text:
            print(f"[DEBUG] No text detected by any OCR configuration")
            return "", 0
            
        print(f"[DEBUG] Best OCR result: '{best_text}' (confidence: {best_confidence:.1f})")
        # Clean and format the extracted text
        equation_text = clean_equation_text(best_text)
        
        return equation_text, best_confidence
        
    except pytesseract.pytesseract.TesseractNotFoundError:
        raise Exception(r"C:\Program Files\Tesseract-OCR\tesseract.exe is not installed or it's not in your PATH. See README file for more information.")
    except Exception as e:
        print(f"[ERROR] OCR processing failed: {e}")
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
        
    print(f"[DEBUG] Cleaning text: '{text}'")
    
    # Remove extra spaces
    text = " ".join(text.split())
    
    import re
    
    # Handle common OCR mistakes and mathematical symbols more carefully
    # Only replace characters that are clearly OCR errors in mathematical context
    
    # Check if text contains function calls that need special handling
    has_functions = any(func in text.lower() for func in ['integrate', 'limit', 'minimize', 'maximize', 'mean', 'std'])
    
    if has_functions:
        # For function calls, be more careful with replacements
        # Only replace safe characters that won't break function syntax
        safe_replacements = {
            '÷': '/',   # Replace division symbol with slash
            '×': '*',   # Replace multiplication symbol with asterisk
            '·': '*',   # Replace middle dot with asterisk
            '^': '**',  # Replace caret with double asterisk for power
            'π': 'pi',  # Replace pi symbol with 'pi'
            '√': 'sqrt',  # Replace square root symbol with 'sqrt'
            '"': '',    # Remove quotes
            "'": '',    # Remove single quotes
            '`': '',    # Remove backticks
            '_': '',    # Remove underscores (often OCR noise)
            '—': '-',   # Replace em-dash with minus
            '–': '-',   # Replace en-dash with minus
            '−': '-',   # Replace Unicode minus with regular minus
            '‐': '-',   # Replace hyphen with minus
            '‑': '-',   # Replace non-breaking hyphen with minus
        }
        
        # Apply only safe replacements for function calls
        for old, new in safe_replacements.items():
            text = text.replace(old, new)
    else:
        # For non-function text, apply all replacements including OCR fixes
        full_replacements = {
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
            'π': 'pi',  # Replace pi symbol with 'pi'
            '√': 'sqrt',  # Replace square root symbol with 'sqrt'
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
            ' ': '',    # Remove spaces
        }
        
        for old, new in full_replacements.items():
            text = text.replace(old, new)
    
    # Add explicit multiplication between numbers and variables
    text = add_explicit_multiplication(text)
    
    # Ensure X is lowercase for consistency
    text = text.replace('X', 'x')
    
    # Remove trailing periods
    text = text.rstrip('.')
    
    return text
    
    # Handle common character confusions more carefully
    # Only replace if it makes sense in mathematical context
    # Replace standalone letters that are likely numbers
    text = re.sub(r'(?<!\w)[Oo](?!\w)', '0', text)  # O -> 0 when standalone
    text = re.sub(r'(?<!\w)[lI](?!\w)', '1', text)   # l/I -> 1 when standalone  
    text = re.sub(r'(?<!\w)[Ss](?!\w)', '5', text)  # S -> 5 when standalone
    text = re.sub(r'(?<!\w)[Zz](?!\w)', '2', text)  # Z -> 2 when standalone
    
    # Handle spaces between numbers and variables (e.g., "x 11" should be "x*11")
    # Add multiplication between variable and number (e.g., "x 2" becomes "x*2")
    text = re.sub(r'([a-zA-Z])\s*(\d)', r'\1*\2', text)
    
    # Add multiplication between number and variable (e.g., "2 x" becomes "2*x")
    text = re.sub(r'(\d)\s*([a-zA-Z])', r'\1*\2', text)
    
    # Handle cases like "2x" -> "2*x" (no space between number and variable)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    
    # Remove any remaining spaces
    text = text.replace(" ", "")
    
    # Ensure 'x' is lowercase for sympy (but preserve function names)
    # Only replace standalone X, not in function names
    text = re.sub(r'\bX\b', 'x', text)
    
    # Remove trailing periods that might be added by OCR
    if text.endswith('.'):
        text = text[:-1]
    
    # Add explicit multiplication operators for implicit multiplication
    text = add_explicit_multiplication(text)
    
    print(f"[DEBUG] Cleaned text result: '{text}'")
    return text

def add_explicit_multiplication(text):
    """
    Add explicit multiplication operators for implicit multiplication.
    Converts patterns like '2x' to '2*x', '3y' to '3*y', etc.
    
    Args:
        text: Equation text that may contain implicit multiplication
        
    Returns:
        Text with explicit multiplication operators
    """
    import re
    
    # Skip if this contains function placeholders or special syntax
    if '__FUNC_' in text or any(func in text.lower() for func in ['limit', 'integrate', 'minimize', 'mean', 'gamma']):
        return text
    
    # Protect function arguments from multiplication insertion
    function_pattern = r'\b(sin|cos|tan|sec|csc|cot|log|ln|sqrt|exp)\s*\([^)]*\)'
    protected_parts = re.findall(function_pattern, text)
    
    # Replace function calls with placeholders (use a more unique placeholder)
    placeholder_map = {}
    for i, func_call in enumerate(protected_parts):
        placeholder = f'___FUNC_PLACEHOLDER_{i}___'
        placeholder_map[placeholder] = func_call
        text = text.replace(func_call, placeholder)
    
    # Handle trigonometric functions with implicit multiplication first
    # Pattern to match: trigonometric function with implicit multiplication like sin(2x), cos(3y)
    trig_pattern = r'\b(sin|cos|tan|sec|csc|cot)\(([^)]*\d)([a-zA-Z])([^)]*)\)'
    
    def fix_trig_multiplication(match):
        func_name = match.group(1)
        before = match.group(2)
        var = match.group(3)
        after = match.group(4)
        return f"{func_name}({before}*{var}{after})"
    
    text = re.sub(trig_pattern, fix_trig_multiplication, text)
    
    # Pattern to match: digit(s) followed immediately by letter(s) but not function names
    # This handles cases like 2x, 3y, 4z, 5a, etc. but not 2sin(x)
    # Use negative lookahead to avoid matching function names
    pattern = r'(\d+)([a-zA-Z])(?!\w*\()'
    
    # Replace with digit * letter
    text = re.sub(pattern, r'\1*\2', text)
    
    # Also handle cases where there's a closing parenthesis followed by a variable
    # like (x+1)x should become (x+1)*x
    pattern2 = r'(\))(\w+)(?!\w*\()'
    text = re.sub(pattern2, r'\1*\2', text)
    
    # Restore protected function calls before applying pattern3
    for placeholder, func_call in placeholder_map.items():
        text = text.replace(placeholder, func_call)
    
    # Handle cases where a variable is followed by an opening parenthesis
    # like x(x+1) should become x*(x+1), but not sin(x) which should stay as sin(x)
    # Use word boundary to ensure we match complete words, and exclude function names
    pattern3 = r'\b(?!sin|cos|tan|log|ln|sqrt|exp|factorial|gamma)([a-zA-Z]\w*)\b(\()(?!\w*\()'
    text = re.sub(pattern3, r'\1*\2', text)
    
    return text

def solve_equation(equation_text, model):
    """
    Parse and solve the equation using enhanced equation solver with fallback to Gemini.
    
    Args:
        equation_text: Cleaned equation text
        model: The GenerativeModel instance for Gemini fallback
        
    Returns:
        tuple: (result, error_message)
    """
    logger.debug(f"solve_equation called with: {equation_text}")
    
    if not equation_text or equation_text.isspace():
        logger.debug("No equation detected")
        return None, "No equation detected"

    # Try enhanced solver first
    logger.debug("Trying enhanced solver...")
    try:
        from equation_solver import solve_equation_enhanced
        result, error = solve_equation_enhanced(equation_text)
        logger.debug(f"Enhanced solver result: {result}, error: {error}")
        if result is not None or error is None:
            return result, error
    except ImportError as e:
        logger.debug(f"Enhanced solver not available: {e}")
        # Enhanced solver not available, continue with original implementation
        pass
    except Exception as e:
        logger.debug(f"Enhanced solver failed: {e}")
        # Enhanced solver failed, fallback to original implementation
        pass

    # Original sympy-based solver
    try:
        # Import additional functions for advanced math support
        from sympy import sin, cos, tan, sqrt, log, ln, pi, E, expand, diff, integrate, limit, oo, factorial, Matrix, Function
        from sympy.abc import a, b, c, d, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w

        # Define symbols
        x, y, z = symbols('x y z')
        t = symbols('t')

        # Check for system of equations (multiple '=' signs)
        if equation_text.count('=') > 1 and ';' in equation_text:
            # Split into individual equations
            equations = equation_text.split(';')
            equations = [eq.strip() for eq in equations if eq.strip()]

            try:
                # Parse each equation
                parsed_equations = []
                all_variables = set()

                for eq in equations:
                    if '=' in eq:
                        left_side, right_side = eq.split('=')
                        left_expr = sympify(left_side.strip())
                        right_expr = sympify(right_side.strip())
                        parsed_eq = Eq(left_expr, right_expr)
                        parsed_equations.append(parsed_eq)

                        # Collect all variables
                        for var in [x, y, z]:
                            if var in parsed_eq.free_symbols:
                                all_variables.add(var)

                if not all_variables:
                    return None, "No variables found in system of equations"

                # Solve the system of equations
                solution = solve(parsed_equations, list(all_variables))

                if solution:
                    return solution, None
                else:
                    return None, "Could not solve the system of equations"
            except Exception as e:
                return None, f"Error solving system of equations: {str(e)}"

        # Handle differential equations
        elif 'diff(' in equation_text or 'd/dx' in equation_text or 'd/dt' in equation_text:
            try:
                # Define a function for dsolve
                from sympy.solvers.ode import dsolve

                # Pre-process derivative notation
                if 'd/dx' in equation_text:
                    equation_text = equation_text.replace('d/dx', 'diff(f(x),x)')
                elif 'd/dt' in equation_text:
                    equation_text = equation_text.replace('d/dt', 'diff(f(t),t)')

                # Define the function to solve for
                f = symbols('f', cls=Function)

                # Parse the equation
                if '=' in equation_text:
                    left_side, right_side = equation_text.split('=')
                    left_expr = sympify(left_side.strip())
                    right_expr = sympify(right_side.strip())
                    diff_eq = Eq(left_expr, right_expr)
                else:
                    diff_eq = sympify(equation_text)

                solution = dsolve(diff_eq, f(x) if 'x' in equation_text else f(t))
                return solution, None
            except Exception as e:
                return None, f"Error solving differential equation: {str(e)}"

        # Handle integrals
        elif 'integrate(' in equation_text:
            try:
                expr_str = equation_text.replace('integrate(', '').replace(')', '')
                expr = sympify(expr_str)
                integral_result = integrate(expr, x)
                return integral_result, None
            except Exception as e:
                return None, f"Error solving integral: {str(e)}"

        # Handle limits
        elif 'limit(' in equation_text:
            try:
                # Example: limit(1/x, x, 0)
                parts = equation_text.replace('limit(', '').replace(')', '').split(',')
                expr_str = parts[0].strip()
                var_str = parts[1].strip()
                point_str = parts[2].strip()

                expr = sympify(expr_str)
                var = symbols(var_str)
                point = sympify(point_str)

                limit_result = limit(expr, var, point)
                return limit_result, None
            except Exception as e:
                return None, f"Error solving limit: {str(e)}"

        # Handle matrix equations
        elif 'Matrix(' in equation_text:
            try:
                matrix_expr = sympify(equation_text)
                return matrix_expr, None
            except Exception as e:
                return None, f"Error solving matrix equation: {str(e)}"

        # General algebraic equation or expression
        elif '=' in equation_text:
            left_side, right_side = equation_text.split('=')
            lhs = sympify(left_side.strip())
            rhs = sympify(right_side.strip())
            equation = Eq(lhs, rhs)
            solution = solve(equation, x)  # Assuming 'x' is the variable to solve for
            if solution:
                return solution, None
            else:
                return None, "Could not solve the equation"
        else:
            # Evaluate mathematical expression
            try:
                result = sympify(equation_text)
                return result, None
            except SympifyError:
                # Fallback to Gemini for complex or unparseable equations
                try:
                    prompt = f"Solve the following mathematical equation or expression: {equation_text}. Provide only the solution, no extra text."
                    response = model.generate_content(prompt)
                    return response.text, None
                except Exception as gemini_e:
                    return None, f"Gemini fallback failed: {str(gemini_e)}"

    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}"