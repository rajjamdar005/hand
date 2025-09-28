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
    2. Resize (3x upscaling for better detail)
    3. Apply noise reduction
    4. Apply adaptive thresholding
    5. Apply morphological operations to enhance text
    6. Apply dilation to connect broken strokes
    
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
    
    # Resize (3x upscaling for better detail)
    height, width = gray.shape
    gray = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter for noise reduction while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply adaptive thresholding to get a binary image
    binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Apply morphological operations to enhance text
    kernel = np.ones((2, 2), np.uint8)
    
    # Opening operation to remove small noise
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilation to connect broken strokes
    dilation = cv2.dilate(opening, kernel, iterations=2) # Increased iterations for better connection
    
    # Save the preprocessed image
    preprocessed_path = os.path.splitext(image_path)[0] + '_preprocessed.png'
    cv2.imwrite(preprocessed_path, dilation)
    
    return preprocessed_path

def extract_equation(image_path):
    """
    Extract equation text from the preprocessed image using OCR
    
    Args:
        image_path: Path to the preprocessed image
        
    Returns:
        tuple: (equation_text, confidence)
    """
    # OCR configuration for math equations - expanded character set
    custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist="0123456789+-*/()=xXyYzZ^.√πesin cos tan log ln" '
    
    # Define a confidence threshold
    CONFIDENCE_THRESHOLD = 0  # Lowered to accept all valid text including confidence 0

    try:
        # Perform OCR
        ocr_data = pytesseract.image_to_data(cv2.imread(image_path), config=custom_config, output_type=pytesseract.Output.DICT)
        logger.debug(f"OCR Data: {ocr_data}") # Debug log
    except pytesseract.pytesseract.TesseractNotFoundError:
        raise Exception(r"C:\Program Files\Tesseract-OCR\tesseract.exe is not installed or it's not in your PATH. See README file for more information.")
    
    # Extract text and confidence, filtering by confidence threshold
    texts = [word for word, conf in zip(ocr_data['text'], ocr_data['conf']) if word.strip()]
    confidences = [conf for word, conf in zip(ocr_data['text'], ocr_data['conf']) if word.strip()]
    logger.debug(f"Filtered Texts: {texts}") # Debug log
    logger.debug(f"Filtered Confidences: {confidences}") # Debug log
    
    if not texts:
        logger.debug("No texts detected after filtering.") # Debug log
        return "", 0
    
    # Join all detected text parts
    equation_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    logger.debug(f"Joined Equation Text (before cleaning): {equation_text}") # Debug log
    
    # Clean and format the extracted text
    equation_text = clean_equation_text(equation_text)
    logger.debug(f"Cleaned Equation Text: {equation_text}") # Debug log
    
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
    
    import re
    
    # Handle common OCR mistakes and mathematical symbols more carefully
    # Only replace characters that are clearly OCR errors in mathematical context
    
    # Replace specific mathematical symbols
    replacements = {
        '÷': '/',  # Replace division symbol with slash
        '×': '*',  # Replace multiplication symbol with asterisk
        '^': '**',  # Replace caret with double asterisk for power
        'π': 'pi',  # Replace pi symbol with 'pi'
        '√': 'sqrt',  # Replace square root symbol with 'sqrt'
        '"': '',   # Remove quotes
        "'": '',   # Remove single quotes
        '_': '-',    # Replace underscore with hyphen
        ' ': '',     # Remove spaces
    }
    
    # Protect function arguments from comma replacement
    import re
    
    def protect_function_args(text):
        """Protect commas in function arguments, handling nested parentheses"""
        # Pattern to match function calls with their arguments, handling nested parentheses
        # This uses a more sophisticated approach to handle nested structures
        result = text
        
        # Find all potential function calls (word followed by opening parenthesis)
        function_starts = list(re.finditer(r'\b(\w+)\s*\(', text))
        
        # Process from right to left to avoid index shifting issues
        for match in reversed(function_starts):
            func_name = match.group(1)
            start_pos = match.end() - 1  # Position of the opening parenthesis
            
            # Find the matching closing parenthesis
            paren_count = 0
            end_pos = start_pos
            for i in range(start_pos, len(text)):
                if text[i] == '(':
                    paren_count += 1
                elif text[i] == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        end_pos = i
                        break
            
            if end_pos > start_pos:
                # Extract the arguments
                args = text[start_pos + 1:end_pos]
                # Protect commas in the arguments
                protected_args = args.replace(',', '__COMMA__')
                # Replace in the result
                result = result[:start_pos + 1] + protected_args + result[end_pos:]
        
        return result
    
    # Apply protection
    text = protect_function_args(text)
    
    # Only replace comma with dot if it's not in a matrix context
    # Check if this looks like a matrix (contains [[ and ]])
    if not ('[[' in text and ']]' in text):
        text = text.replace(',', '.')  # Replace comma with decimal point only for non-matrix expressions
    
    # Restore protected commas in function arguments
    text = text.replace('__COMMA__', ',')
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Handle common character confusions more carefully
    # Only replace if it makes sense in mathematical context
    # Replace standalone letters that are likely numbers
    text = re.sub(r'(?<!\w)[Oo](?!\w)', '0', text)  # O -> 0 when standalone
    text = re.sub(r'(?<!\w)[lI](?!\w)', '1', text)   # l/I -> 1 when standalone  
    text = re.sub(r'(?<!\w)[Ss](?!\w)', '5', text)  # S -> 5 when standalone
    text = re.sub(r'(?<!\w)[Zz](?!\w)', '2', text)  # Z -> 2 when standalone
    
    # Ensure 'x' is lowercase for sympy (but preserve function names)
    # Only replace standalone X, not in function names
    text = re.sub(r'\bX\b', 'x', text)
    
    # Remove trailing periods that might be added by OCR
    if text.endswith('.'):
        text = text[:-1]
    
    # Add explicit multiplication operators for implicit multiplication
    text = add_explicit_multiplication(text)
    
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