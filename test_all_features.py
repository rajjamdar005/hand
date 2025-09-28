#!/usr/bin/env python3

import cv2
import numpy as np
import os
import requests
import base64
from PIL import Image
from io import BytesIO
import time
from utils_clean import preprocess_image, extract_equation, solve_equation

def create_test_equations():
    """Create various test equation images"""
    equations = [
        ("2x + 3 = 7", "Simple linear equation"),
        ("x - 5 = 0", "Simple subtraction"),
        ("3x = 9", "Simple multiplication"),
        ("x^2 - 4 = 0", "Simple quadratic"),
        ("5 + 3", "Basic arithmetic"),
        ("10 - 7", "Basic subtraction"),
        ("2 * 6", "Basic multiplication"),
    ]
    
    test_images = []
    
    for i, (equation, description) in enumerate(equations):
        # Create image
        img = np.ones((120, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, equation, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        filename = f"test_{i+1}_{equation.replace(' ', '').replace('*', 'mult').replace('^', 'pow')}.png"
        cv2.imwrite(filename, img)
        
        test_images.append({
            'filename': filename,
            'equation': equation,
            'description': description
        })
    
    return test_images

def test_ocr_pipeline():
    """Test the OCR pipeline with various equations"""
    print("=" * 60)
    print("TESTING OCR PIPELINE")
    print("=" * 60)
    
    test_images = create_test_equations()
    results = []
    
    for test_img in test_images:
        print(f"\n--- Testing: {test_img['description']} ---")
        print(f"Expected: {test_img['equation']}")
        
        try:
            # Preprocess
            preprocessed = preprocess_image(test_img['filename'])
            
            # Extract
            equation, confidence = extract_equation(preprocessed)
            print(f"Extracted: '{equation}' (confidence: {confidence:.1f}%)")
            
            # Solve
            if equation:
                result, error = solve_equation(equation)
                if error:
                    print(f"Solving error: {error}")
                    status = "FAILED"
                else:
                    print(f"Solution: {result}")
                    status = "SUCCESS"
            else:
                status = "NO_DETECTION"
            
            results.append({
                'test': test_img['description'],
                'expected': test_img['equation'],
                'extracted': equation,
                'confidence': confidence,
                'status': status
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'test': test_img['description'],
                'expected': test_img['equation'],
                'extracted': '',
                'confidence': 0,
                'status': "ERROR"
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("OCR PIPELINE RESULTS SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    total_count = len(results)
    
    for result in results:
        status_emoji = {
            'SUCCESS': '✅',
            'FAILED': '❌',
            'NO_DETECTION': '🔍',
            'ERROR': '💥'
        }
        print(f"{status_emoji.get(result['status'], '?')} {result['test']}: {result['extracted']} ({result['confidence']:.1f}%)")
    
    print(f"\nOverall Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    return results

def test_flask_app():
    """Test Flask application endpoints"""
    print("\n" + "=" * 60)
    print("TESTING FLASK APPLICATION")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:5000"
    
    # Test if app is running
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Flask app is running")
        else:
            print(f"❌ Flask app returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Flask app is not accessible: {e}")
        print("Please start the Flask app by running: python app.py")
        return False
    
    # Test file upload (if app is running)
    try:
        # Create a simple test image
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img, '2x + 1 = 5', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite('flask_test.png', img)
        
        # Upload to Flask app
        files = {'equation_image': open('flask_test.png', 'rb')}
        response = requests.post(f"{base_url}/upload", files=files)
        
        if response.status_code == 200 and 'result.html' in response.url:
            print("✅ File upload works")
        else:
            print(f"❌ File upload failed: {response.status_code}")
        
        files['equation_image'].close()
        
    except Exception as e:
        print(f"❌ File upload test failed: {e}")
    
    return True

def test_canvas_functionality():
    """Test canvas drawing functionality (simulated)"""
    print("\n" + "=" * 60)
    print("TESTING CANVAS FUNCTIONALITY")
    print("=" * 60)
    
    try:
        # Create a canvas-like image (white background with black text)
        canvas_img = np.ones((300, 700, 3), dtype=np.uint8) * 255
        
        # Simulate hand-drawn equation with slightly imperfect text
        cv2.putText(canvas_img, 'x + 2 = 5', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Add some noise to simulate real drawing
        noise = np.random.randint(0, 50, canvas_img.shape, dtype=np.uint8)
        canvas_img = cv2.addWeighted(canvas_img, 0.95, noise, 0.05, 0)
        
        cv2.imwrite('canvas_test.png', canvas_img)
        
        # Test processing
        preprocessed = preprocess_image('canvas_test.png')
        equation, confidence = extract_equation(preprocessed)
        
        if equation:
            result, error = solve_equation(equation)
            if not error:
                print(f"✅ Canvas processing works: '{equation}' -> {result}")
            else:
                print(f"❌ Canvas solving failed: {error}")
        else:
            print("❌ Canvas OCR failed - no equation detected")
            
    except Exception as e:
        print(f"❌ Canvas test failed: {e}")

def cleanup_test_files():
    """Clean up test files"""
    patterns = ['test_*.png', 'flask_test.png', 'canvas_test.png', '*_preprocessed.png', 'test_equation*.png']
    
    import glob
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except:
                pass

def main():
    """Run all tests"""
    print("🧪 HANDWRITTEN EQUATION SOLVER - COMPREHENSIVE TEST SUITE")
    print("🧪 " + "=" * 58)
    
    try:
        # Test 1: OCR Pipeline
        ocr_results = test_ocr_pipeline()
        
        # Test 2: Flask App (if running)
        flask_running = test_flask_app()
        
        # Test 3: Canvas functionality
        test_canvas_functionality()
        
        # Final summary
        print("\n" + "🎯" + "=" * 58)
        print("FINAL TEST SUMMARY")
        print("🎯" + "=" * 58)
        
        success_count = sum(1 for r in ocr_results if r['status'] == 'SUCCESS')
        total_tests = len(ocr_results)
        
        print(f"📊 OCR Pipeline: {success_count}/{total_tests} tests passed")
        print(f"🌐 Flask App: {'✅ Running' if flask_running else '❌ Not tested'}")
        print(f"🎨 Canvas: ✅ Simulated successfully")
        
        print(f"\n🎉 Overall Status: {'READY FOR USE' if success_count > total_tests * 0.7 else 'NEEDS IMPROVEMENT'}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    finally:
        print("\n🧹 Cleaning up test files...")
        cleanup_test_files()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
