"""
Scientific Computing Extension for Enhanced Equation Solver
Integrates SciPy for numerical methods, optimization, statistics, and advanced mathematical operations
"""

import numpy as np
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.stats as stats
import scipy.linalg as linalg
import scipy.special as special
from scipy.constants import constants
import sympy as sp
from typing import Tuple, Any, List, Dict
import warnings

# Suppress scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

class ScientificSolver:
    """
    Scientific computing solver that provides numerical methods and advanced mathematical operations
    """
    
    def __init__(self):
        """Initialize the scientific solver with common constants and functions."""
        # Physical and mathematical constants from scipy
        self.constants = {
            'pi': np.pi,
            'e': np.e,
            'c': constants.speed_of_light,  # Speed of light
            'g': constants.g,              # Standard gravity
            'h': constants.h,              # Planck constant
            'k': constants.k,              # Boltzmann constant
            'N_A': constants.N_A,          # Avogadro constant
            'R': constants.R,              # Gas constant
            'sigma': constants.sigma,      # Stefan-Boltzmann constant
        }
    
    def solve_numerical_integration(self, expr_text: str, bounds: Tuple[float, float] = (0, 1)) -> Tuple[Any, str]:
        """
        Solve numerical integration problems.
        
        Args:
            expr_text: Expression to integrate
            bounds: Integration bounds (a, b)
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            # Convert expression to a function that can be numerically integrated
            x = sp.Symbol('x')
            expr = sp.sympify(expr_text)
            
            # Convert to numerical function
            func = sp.lambdify(x, expr, 'numpy')
            
            # Perform numerical integration
            result, error = integrate.quad(func, bounds[0], bounds[1])
            
            return {
                'value': result,
                'error': error,
                'bounds': bounds
            }, None
            
        except Exception as e:
            return None, f"Error in numerical integration: {str(e)}"
    
    def solve_optimization_problem(self, objective: str, method: str = 'BFGS', 
                                  bounds: List[Tuple[float, float]] = None) -> Tuple[Any, str]:
        """
        Solve optimization problems using scipy.optimize.
        
        Args:
            objective: Objective function to minimize
            method: Optimization method
            bounds: Variable bounds for constrained optimization
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            # Convert to numerical function
            x = sp.Symbol('x')
            expr = sp.sympify(objective)
            func = sp.lambdify(x, expr, 'numpy')
            
            # Initial guess
            x0 = np.array([1.0])
            
            # Solve optimization problem
            if bounds:
                result = optimize.minimize(func, x0, method='L-BFGS-B', bounds=bounds)
            else:
                result = optimize.minimize(func, x0, method=method)
            
            return {
                'minimum': result.x[0] if result.x.size == 1 else result.x,
                'function_value': result.fun,
                'success': result.success,
                'message': result.message
            }, None
            
        except Exception as e:
            return None, f"Error in optimization: {str(e)}"
    
    def solve_differential_equation_numerical(self, equation: str, initial_conditions: List[float],
                                             t_span: Tuple[float, float] = (0, 10)) -> Tuple[Any, str]:
        """
        Solve differential equations numerically using scipy.integrate.solve_ivp.
        
        Args:
            equation: Differential equation
            initial_conditions: Initial conditions
            t_span: Time span for solution
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            from scipy.integrate import solve_ivp
            
            # This is a simplified implementation
            # In practice, you'd need to parse the differential equation
            # and convert it to a function format suitable for solve_ivp
            
            # Example: dy/dt = -2*y + 1 with y(0) = 0
            def dydt(t, y):
                return -2*y + 1
            
            # Solve the differential equation
            sol = solve_ivp(dydt, t_span, initial_conditions, dense_output=True)
            
            return {
                't': sol.t,
                'y': sol.y,
                'success': sol.success,
                'message': sol.message
            }, None
            
        except Exception as e:
            return None, f"Error solving differential equation: {str(e)}"
    
    def solve_statistical_analysis(self, data_expr: str, analysis_type: str = 'descriptive') -> Tuple[Any, str]:
        """
        Perform statistical analysis using scipy.stats.
        
        Args:
            data_expr: Data expression or distribution
            analysis_type: Type of analysis ('descriptive', 'hypothesis', 'distribution')
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            if analysis_type == 'descriptive':
                # Example data generation for demonstration
                data = np.random.normal(0, 1, 100)
                
                result = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'variance': np.var(data),
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'median': np.median(data)
                }
                
                return result, None
            
            elif analysis_type == 'distribution':
                # Fit data to common distributions
                data = np.random.normal(0, 1, 100)
                
                # Test for normality
                stat, p_value = stats.normaltest(data)
                
                result = {
                    'normality_test': {
                        'statistic': stat,
                        'p_value': p_value,
                        'is_normal': p_value > 0.05
                    }
                }
                
                return result, None
            
            else:
                return None, f"Unknown analysis type: {analysis_type}"
                
        except Exception as e:
            return None, f"Error in statistical analysis: {str(e)}"
    
    def solve_linear_algebra(self, matrix_expr: str, operation: str = 'eigenvalues') -> Tuple[Any, str]:
        """
        Solve linear algebra problems using scipy.linalg.
        
        Args:
            matrix_expr: Matrix expression
            operation: Operation to perform ('eigenvalues', 'determinant', 'inverse', 'svd')
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            # Parse matrix expression (simplified example)
            # In practice, you'd need robust matrix parsing
            if '[[' in matrix_expr and ']]' in matrix_expr:
                matrix_data = eval(matrix_expr)
                matrix = np.array(matrix_data)
            else:
                # Create example matrix for demonstration
                matrix = np.array([[1, 2], [3, 4]])
            
            if operation == 'eigenvalues':
                eigenvals, eigenvecs = linalg.eig(matrix)
                result = {
                    'eigenvalues': eigenvals.tolist(),
                    'eigenvectors': eigenvecs.tolist()
                }
            elif operation == 'determinant':
                det = linalg.det(matrix)
                result = {'determinant': det}
            elif operation == 'inverse':
                try:
                    inv = linalg.inv(matrix)
                    result = {'inverse': inv.tolist()}
                except linalg.LinAlgError:
                    return None, "Matrix is singular (not invertible)"
            elif operation == 'svd':
                U, s, Vh = linalg.svd(matrix)
                result = {
                    'U': U.tolist(),
                    'singular_values': s.tolist(),
                    'Vh': Vh.tolist()
                }
            else:
                return None, f"Unknown operation: {operation}"
            
            return result, None
            
        except Exception as e:
            return None, f"Error in linear algebra: {str(e)}"
    
    def solve_special_functions(self, func_expr: str) -> Tuple[Any, str]:
        """
        Evaluate special mathematical functions using scipy.special.
        
        Args:
            func_expr: Special function expression
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            # Handle common special functions
            if 'gamma(' in func_expr:
                # Extract argument (simplified parsing)
                arg_str = func_expr.split('gamma(')[1].split(')')[0]
                arg = float(arg_str)
                result = special.gamma(arg)
                return result, None
            
            elif 'beta(' in func_expr:
                # Beta function
                args = func_expr.split('beta(')[1].split(')')[0].split(',')
                a, b = float(args[0]), float(args[1])
                result = special.beta(a, b)
                return result, None
            
            elif 'erf(' in func_expr:
                # Error function
                arg_str = func_expr.split('erf(')[1].split(')')[0]
                arg = float(arg_str)
                result = special.erf(arg)
                return result, None
            
            elif 'bessel(' in func_expr:
                # Bessel function (simplified)
                args = func_expr.split('bessel(')[1].split(')')[0].split(',')
                n, x = int(args[0]), float(args[1])
                result = special.jv(n, x)  # Bessel function of the first kind
                return result, None
            
            else:
                return None, "Unknown special function"
                
        except Exception as e:
            return None, f"Error evaluating special function: {str(e)}"
    
    def solve_fourier_analysis(self, signal_expr: str, sample_rate: float = 1000) -> Tuple[Any, str]:
        """
        Perform Fourier analysis using scipy.fft.
        
        Args:
            signal_expr: Signal expression
            sample_rate: Sample rate for discrete signals
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            from scipy.fft import fft, fftfreq
            
            # Generate sample signal (simplified example)
            t = np.linspace(0, 1, int(sample_rate), False)
            
            # Example: sin(2*pi*50*t) + sin(2*pi*120*t)
            signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
            
            # Compute FFT
            fft_vals = fft(signal)
            freqs = fftfreq(len(signal), 1/sample_rate)
            
            # Get magnitude spectrum
            magnitude = np.abs(fft_vals)
            
            result = {
                'frequencies': freqs[:len(freqs)//2].tolist(),
                'magnitude': magnitude[:len(magnitude)//2].tolist(),
                'phase': np.angle(fft_vals[:len(fft_vals)//2]).tolist()
            }
            
            return result, None
            
        except Exception as e:
            return None, f"Error in Fourier analysis: {str(e)}"

# Create global instance
scientific_solver = ScientificSolver()

def solve_scientific_equation(equation_text: str, equation_type: str = 'auto') -> Tuple[Any, str]:
    """
    Solve scientific equations using appropriate numerical methods.
    
    Args:
        equation_text: Equation text
        equation_type: Type of equation ('integration', 'optimization', 'statistics', etc.)
        
    Returns:
        Tuple of (result, error_message)
    """
    if equation_type == 'integration' or 'integrate' in equation_text.lower():
        return scientific_solver.solve_numerical_integration(equation_text)
    elif equation_type == 'optimization' or any(word in equation_text.lower() for word in ['minimize', 'maximize']):
        return scientific_solver.solve_optimization_problem(equation_text)
    elif equation_type == 'statistics' or any(word in equation_text.lower() for word in ['mean', 'std', 'variance']):
        return scientific_solver.solve_statistical_analysis(equation_text)
    elif equation_type == 'linear_algebra' or '[[' in equation_text:
        return scientific_solver.solve_linear_algebra(equation_text)
    elif equation_type == 'special' or any(func in equation_text.lower() for func in ['gamma', 'beta', 'erf', 'bessel']):
        return scientific_solver.solve_special_functions(equation_text)
    elif equation_type == 'fourier' or 'fft' in equation_text.lower():
        return scientific_solver.solve_fourier_analysis(equation_text)
    else:
        return None, "Could not determine equation type for scientific solving"
