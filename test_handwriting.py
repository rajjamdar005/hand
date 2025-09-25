#!/usr/bin/env python3

import cv2
import numpy as np
import os
from utils_enhanced import preprocess_image, extract_equation, solve_equation

def create_handwritten_test_images():
    """Create various handwritten-style test images"""
    tests = [
        ("1+2", "Simple addition on dark background"),
        ("3x=6", "Simple equation on dark background"),
        ("2*5", "Multiplication on dark background"),
        ("x-1=4", "Subtraction equation on dark background"),
    ]
    
    results = []
    
    for i, (equation, description) in enumerate(tests):
        print(f"\n=== Test {i+1}: {description} ===")
        print(f"Expected: {equation}")
        
        # Create dark background image with white text (like chalk on blackboard)
        img = np.zeros((300, 600, 3), dtype=np.uint8)
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4
        thickness = 8
        
        text_size = cv2.getTextSize(equation, font, font_scale, thickness)[0]
        x = (600 - text_size[0]) // 2
        y = (300 + text_size[1]) // 2
        
        # Draw white text on dark background
        cv2.putText(img, equation, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        filename = f"handwritten_test_{i+1}.png"
        cv2.imwrite(filename, img)
        
        # Test the pipeline
        try:
            preprocessed_paths = preprocess_image(filename)
            extracted, confidence = extract_equation(preprocessed_paths)
            
            print(f"Extracted: '{extracted}' (confidence: {confidence:.1f}%)")
            
            if extracted:
                result, error = solve_equation(extracted)
                if error:
                    print(f"Solving error: {error}")
                    status = "EXTRACTION_OK_SOLVING_FAILED"
                else:
                    print(f"Solution: {result}")
                    status = "SUCCESS"
            else:
                print("No equation detected")
                status = "NO_DETECTION"
                
            results.append({
                'expected': equation,
                'extracted': extracted,
                'confidence': confidence,
                'status': status,
                'description': description
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'expected': equation,
                'extracted': '',
                'confidence': 0,
                'status': "ERROR",
                'description': description
            })
    
    return results

def test_flask_integration():
    """Test Flask integration if server is running"""
    print("\n" + "="*60)
    print("TESTING FLASK INTEGRATION")
    print("="*60)
    
    try:
        import requests
        
        # Test if Flask app is running
        response = requests.get("http://127.0.0.1:5000", timeout=3)
        if response.status_code == 200:
            print("✅ Flask app is running")
            
            # Create a test image for upload
            img = np.zeros((300, 500, 3), dtype=np.uint8)
            cv2.putText(img, '1+2', (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)
            cv2.imwrite('flask_upload_test.png', img)
            
            # Test file upload
            files = {'equation_image': open('flask_upload_test.png', 'rb')}
            response = requests.post("http://127.0.0.1:5000/upload", files=files)
            files['equation_image'].close()
            
            if response.status_code == 200:
                print("✅ File upload successful")
                if '1+2' in response.text or '14+2' in response.text:
                    print("✅ OCR processing worked")
                else:
                    print("❌ OCR didn't detect expected equation")
            else:
                print(f"❌ File upload failed: {response.status_code}")
                
        else:
            print(f"❌ Flask app returned status {response.status_code}")
            
    except requests.exceptions.RequestException:
        print("❌ Flask app is not running")
        print("To test Flask integration, run: python app.py")
    except Exception as e:
        print(f"❌ Error testing Flask: {e}")

def main():
    print("🔬 HANDWRITTEN EQUATION RECOGNITION TEST")
    print("🔬" + "="*58)
    
    # Test OCR pipeline
    results = create_handwritten_test_images()
    
    # Summary
    print("\n" + "="*60)
    print("HANDWRITING RECOGNITION RESULTS")
    print("="*60)
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    total_count = len(results)
    
    status_symbols = {
        'SUCCESS': '✅',
        'NO_DETECTION': '🔍', 
        'EXTRACTION_OK_SOLVING_FAILED': '⚠️',
        'ERROR': '❌'
    }
    
    for result in results:
        symbol = status_symbols.get(result['status'], '?')
        print(f"{symbol} {result['description']}")
        print(f"   Expected: '{result['expected']}' | Got: '{result['extracted']}' ({result['confidence']:.1f}%)")
    
    print(f"\n📊 Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    # Test Flask integration
    test_flask_integration()
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    import glob
    for pattern in ['handwritten_test_*.png', '*_preprocessed_*.png', 'flask_upload_test.png']:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except:
                pass
    print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
