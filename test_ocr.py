#!/usr/bin/env python3

import cv2
import numpy as np
from utils import preprocess_image, extract_equation, solve_equation

def test_ocr():
    print("=== OCR Pipeline Test ===")
    
    # Create a simple test image with text
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(img, '2x - 3 = -7', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite('test_equation.png', img)
    
    print("Created test image: test_equation.png")
    
    # Test the pipeline
    try:
        print("\n--- Preprocessing ---")
        preprocessed = preprocess_image('test_equation.png')
        
        print("\n--- OCR Extraction ---")
        equation, confidence = extract_equation(preprocessed)
        
        print(f"\n--- Results ---")
        print(f"Extracted equation: '{equation}'")
        print(f"Confidence: {confidence:.1f}%")
        
        if equation:
            print("\n--- Solving ---")
            result, error = solve_equation(equation)
            if error:
                print(f"Solving error: {error}")
            else:
                print(f"Solution: {result}")
        
        return equation, confidence
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return None, 0

if __name__ == "__main__":
    test_ocr()
