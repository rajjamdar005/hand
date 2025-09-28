#!/usr/bin/env python3
"""
Comprehensive test script for the enhanced equation solver
"""

from equation_solver import solve_equation_enhanced
import json

def test_equation(equation, expected_type=None):
    """Test a single equation and return results"""
    print(f"\n{'='*50}")
    print(f"Testing: {equation}")
    
    result, error = solve_equation_enhanced(equation)
    
    if error:
        print(f"❌ ERROR: {error}")
        return False
    else:
        print(f"✅ SUCCESS:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")
        return True

def main():
    """Run comprehensive tests"""
    print("🧪 Enhanced Equation Solver Test Suite")
    print("="*60)
    
    # Test cases organized by category
    test_cases = [
        # Basic Algebra
        ("2x + 3 = 7", "algebra"),
        ("x^2 - 5x + 6 = 0", "quadratic"),
        ("x^3 - 6x^2 + 11x - 6 = 0", "cubic"),
        
        # Trigonometric
        ("sin(x) = 0.5", "trigonometric"),
        ("cos(2x) = 0.8", "trigonometric"),
        
        # Special Functions
        ("gamma(5)", "special_function"),
        ("factorial(5)", "special_function"),
        
        # Matrices
        ("[[1,2],[3,4]]", "matrix"),
        ("[[1,0],[0,1]]", "matrix"),
        
        # Integration
        ("integrate(x^2, x)", "integration"),
        ("integrate(sin(x), x)", "integration"),
        
        # Limits
        ("limit(sin(x)/x, x, 0)", "limit"),
        ("x^2 -> 2", "limit"),
        
        # Optimization
        ("minimize(x^2 + 2x + 1)", "optimization"),
        ("maximize(-x^2 + 4x)", "optimization"),
        
        # Statistics
        ("mean([1,2,3,4,5])", "statistics"),
        ("std([1,2,3,4,5])", "statistics"),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for equation, expected_type in test_cases:
        try:
            if test_equation(equation, expected_type):
                passed += 1
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} tests failed")

if __name__ == "__main__":
    main()