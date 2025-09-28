"""
Enhanced Equation Solver Module
Integrates multiple mathematical libraries for comprehensive equation solving:
- SymPy for symbolic mathematics
- NumPy for numerical computations
- SciPy for advanced scientific computing
- Support for various equation types including statistical, optimization, and engineering formulas
"""

import numpy as np
import sympy as sp
from sympy import symbols, Eq, solve, sympify, SympifyError, Matrix
from sympy import sin, cos, tan, sqrt, log, ln, pi, E, expand, diff, integrate, limit, oo, factorial, exp, Min, Max, Piecewise
from sympy import Abs as sp_abs, symbols as sp_symbols
from sympy.abc import a, b, c, d, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w
import re
from typing import Tuple, Union, List, Dict, Any
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class EnhancedEquationSolver:
    """
    Enhanced equation solver that combines multiple mathematical libraries
    for comprehensive equation solving capabilities.
    """
    
    def __init__(self):
        """Initialize the enhanced equation solver with predefined symbols and functions."""
        # Define common symbols
        self.x, self.y, self.z = symbols('x y z')
        self.t = symbols('t')
        self.symbols_dict = {
            'x': self.x, 'y': self.y, 'z': self.z, 't': self.t,
            'a': a, 'b': b, 'c': c, 'd': d, 'f': f, 'g': g, 'h': h,
            'i': i, 'j': j, 'k': k, 'l': l, 'm': m, 'n': n, 'o': o,
            'p': p, 'q': q, 'r': r, 's': s, 'u': u, 'v': v, 'w': w
        }
        
        # Define mathematical functions
        self.functions_dict = {
            'sin': sin, 'cos': cos, 'tan': tan, 'sqrt': sqrt,
            'log': log, 'ln': ln, 'pi': pi, 'e': E, 'exp': exp,
            'abs': sp_abs, 'diff': diff, 'integrate': integrate,
            'limit': limit, 'factorial': factorial, 'min': Min, 'max': Max
        }
    
    # Removed clean_equation_text from here, it will be called from utils.py
    
    def solve_algebraic_equation(self, equation_text: str) -> Tuple[Any, str]:
        """
        Solve algebraic equations with enhanced support.
        
        Args:
            equation_text: Cleaned equation text
            
        Returns:
            Tuple of (result, error_message)
        """
        logger.debug(f"Algebraic Solver - Received equation: {equation_text}")
        try:
            # Split into left and right sides
            left_side, right_side = equation_text.split('=')
            logger.debug(f"Algebraic Solver - Left side: {left_side}, Right side: {right_side}")
            
            # Parse expressions
            left_expr = sympify(left_side.strip(), locals=self.symbols_dict)
            right_expr = sympify(right_side.strip(), locals=self.symbols_dict)
            equation = Eq(left_expr, right_expr)
            logger.debug(f"Algebraic Solver - SymPy Equation: {equation}")
            
            # Find variables to solve for
            variables = [var for var in [self.x, self.y, self.z, self.t] 
                        if var in equation.free_symbols]
            logger.debug(f"Algebraic Solver - Variables found: {variables}")
            
            if not variables:
                # Try to evaluate as a numerical expression
                if left_expr == right_expr:
                    return "True (Identity)", None
                else:
                    return "False (Contradiction)", None
            
            # Solve for the primary variable
            solution = solve(equation, variables[0])
            logger.debug(f"Algebraic Solver - Solution: {solution}")
            
            if solution:
                return solution, None
            else:
                return None, "Could not find a solution"
                
        except Exception as e:
            logger.error(f"Algebraic Solver - Error: {str(e)}")
            return None, f"Error solving algebraic equation: {str(e)}"
    
    def solve_system_of_equations(self, equations_text: str) -> Tuple[Any, str]:
        """
        Solve system of equations.
        
        Args:
            equations_text: Multiple equations separated by semicolons
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            equations = equations_text.split(';')
            equations = [eq.strip() for eq in equations if eq.strip()]
            
            parsed_equations = []
            all_variables = set()
            
            for eq in equations:
                if '=' in eq:
                    left_side, right_side = eq.split('=')
                    left_expr = sympify(left_side.strip(), locals=self.symbols_dict)
                    right_expr = sympify(right_side.strip(), locals=self.symbols_dict)
                    parsed_eq = Eq(left_expr, right_expr)
                    parsed_equations.append(parsed_eq)
                    
                    # Collect variables
                    for var in [self.x, self.y, self.z, self.t]:
                        if var in parsed_eq.free_symbols:
                            all_variables.add(var)
            
            if not all_variables:
                return None, "No variables found in system"
            
            solution = solve(parsed_equations, list(all_variables))
            
            if solution:
                return solution, None
            else:
                return None, "Could not solve the system of equations"
                
        except Exception as e:
            return None, f"Error solving system: {str(e)}"
    
    def solve_differential_equation(self, equation_text: str) -> Tuple[Any, str]:
        """
        Solve differential equations.
        
        Args:
            equation_text: Differential equation text
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            from sympy.solvers.ode import dsolve
            
            # Handle common differential equation notations
            if 'd/dx' in equation_text:
                equation_text = self._process_derivative_notation(equation_text, 'x')
            elif 'd/dt' in equation_text:
                equation_text = self._process_derivative_notation(equation_text, 't')
            
            if '=' in equation_text:
                left_side, right_side = equation_text.split('=')
                left_expr = sympify(left_side.strip(), locals=self.symbols_dict)
                right_expr = sympify(right_side.strip(), locals=self.symbols_dict)
                diff_eq = Eq(left_expr, right_expr)
                
                # Find the function to solve for
                variables = [var for var in [self.x, self.y, self.z, self.t] 
                           if var in diff_eq.free_symbols]
                
                if not variables:
                    return None, "No variables found in differential equation"
                
                # Attempt to solve
                solution = dsolve(diff_eq, variables[0])
                return solution, None
            else:
                # Just evaluate the expression
                result = sympify(equation_text, locals=self.symbols_dict)
                return result, None
                
        except Exception as e:
            return None, f"Unable to solve differential equation: {str(e)}"
    
    def solve_matrix_equation(self, equation_text: str) -> Tuple[Any, str]:
        """
        Solve matrix equations.
        
        Args:
            equation_text: Matrix equation text
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            # Handle matrix notation like [[1,2],[3,4]]
            if '[[' in equation_text and ']]' in equation_text:
                if '=' in equation_text:
                    left_side, right_side = equation_text.split('=')
                    left_matrix = self._parse_matrix(left_side.strip())
                    right_matrix = self._parse_matrix(right_side.strip())
                    
                    if left_matrix is not None and right_matrix is not None:
                        # Solve matrix equation A*X = B -> X = A^(-1)*B
                        if 'X' in equation_text or 'x' in equation_text:
                            try:
                                solution = left_matrix.inv() * right_matrix
                                return solution, None
                            except:
                                return None, "Matrix is not invertible"
                        else:
                            # Just compare matrices
                            return left_matrix.equals(right_matrix), None
                else:
                    # Just evaluate the matrix expression
                    matrix = self._parse_matrix(equation_text)
                    if matrix is not None:
                        return matrix.tolist(), None
            else:
                # Try to parse as a simple matrix without brackets
                try:
                    # Check if it's a simple matrix expression
                    matrix = self._parse_matrix(equation_text)
                    if matrix is not None:
                        return matrix.tolist(), None
                except:
                    pass
            
            return None, "Invalid matrix format"
            
        except Exception as e:
            return None, f"Error solving matrix equation: {str(e)}"
    
    def solve_optimization_problem(self, equation_text: str) -> Tuple[Any, str]:
        """
        Solve optimization problems (minimize/maximize).
        
        Args:
            equation_text: Optimization problem text
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            # Handle minimize/maximize keywords
            if 'minimize' in equation_text.lower() or 'min' in equation_text.lower():
                objective = equation_text.lower().replace('minimize', '').replace('min', '').strip()
                return self._solve_optimization(objective, 'min'), None
            elif 'maximize' in equation_text.lower() or 'max' in equation_text.lower():
                objective = equation_text.lower().replace('maximize', '').replace('max', '').strip()
                return self._solve_optimization(objective, 'max'), None
            else:
                return None, "No optimization keyword found"
                
        except Exception as e:
            return None, f"Error solving optimization problem: {str(e)}"
    
    def solve_statistical_equation(self, equation_text: str) -> Tuple[Any, str]:
        """
        Solve statistical equations and formulas.
        
        Args:
            equation_text: Statistical equation text
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            # Handle common statistical formulas
            if 'mean' in equation_text or 'average' in equation_text:
                return self._solve_statistical_formula(equation_text, 'mean'), None
            elif 'variance' in equation_text or 'var' in equation_text:
                return self._solve_statistical_formula(equation_text, 'variance'), None
            elif 'std' in equation_text or 'stdev' in equation_text:
                return self._solve_statistical_formula(equation_text, 'std'), None
            else:
                # Try to evaluate as a regular equation
                return sympify(equation_text, locals=self.symbols_dict), None
                
        except Exception as e:
            return None, f"Error solving statistical equation: {str(e)}"
    
    def solve_integration_problem(self, equation_text: str) -> Tuple[Any, str]:
        """
        Solve integration problems.
        
        Args:
            equation_text: Integration problem text
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            # Extract function from integrate(...) format
            if equation_text.lower().startswith('integrate('):
                # Remove 'integrate(' from start and ')' from end
                content = equation_text[10:].rstrip(')').strip()  # Remove 'integrate('
                
                # Handle definite integrals with bounds
                if 'from' in content and 'to' in content:
                    # Format: function from a to b
                    parts = content.split('from')
                    func_part = parts[0].strip()
                    bounds_part = parts[1].replace('to', ',').strip()
                    a, b = bounds_part.split(',')
                    
                    # Parse function and bounds
                    func = sympify(func_part, locals=self.symbols_dict)
                    a_val = sympify(a.strip())
                    b_val = sympify(b.strip())
                    
                    # Compute definite integral
                    result = integrate(func, (self.x, a_val, b_val))
                    return result, None
                else:
                    # Indefinite integral - handle cases like "x**2,x" or "x**2"
                    # If there's a comma, take only the first part (the function)
                    if ',' in content:
                        func_part = content.split(',')[0].strip()
                    else:
                        func_part = content.strip()
                    
                    func = sympify(func_part, locals=self.symbols_dict)
                    result = integrate(func, self.x)
                    return result, None
            else:
                # Handle direct function input (without integrate wrapper)
                func = sympify(equation_text, locals=self.symbols_dict)
                result = integrate(func, self.x)
                return result, None
                
        except Exception as e:
            return None, f"Error solving integration problem: {str(e)}"
    
    def solve_limit_problem(self, equation_text: str) -> Tuple[Any, str]:
        """
        Solve limit problems.
        
        Args:
            equation_text: Limit problem text
            
        Returns:
            Tuple of (result, error_message)
        """
        try:
            # Handle different limit formats
            if 'limit(' in equation_text and ')' in equation_text:
                # Format: limit(function, variable, point)
                content = equation_text.replace('limit(', '').replace(')', '')
                parts = [part.strip() for part in content.split(',')]
                
                if len(parts) == 3:
                    func_str = parts[0].strip()
                    var_str = parts[1].strip()
                    point_str = parts[2].strip()
                    
                    # Handle infinity
                    if point_str.lower() in ['inf', 'infinity', '∞']:
                        point_str = 'oo'
                    elif point_str.lower() in ['-inf', '-infinity', '-∞']:
                        point_str = '-oo'
                    
                    # Parse the function, variable, and point
                    func = sympify(func_str, locals=self.symbols_dict)
                    var = sp_symbols(var_str)
                    point = sympify(point_str, locals=self.symbols_dict)
                    
                    result = limit(func, var, point)
                    return result, None
                else:
                    return None, "Invalid limit format. Use: limit(function, variable, point)"
            else:
                # Try to parse as a simple expression with -> notation
                if '->' in equation_text:
                    parts = equation_text.split('->')
                    if len(parts) == 2:
                        func_str = parts[0].strip()
                        point_str = parts[1].strip()
                        
                        # Handle infinity
                        if point_str.lower() in ['inf', 'infinity', '∞']:
                            point_str = 'oo'
                        elif point_str.lower() in ['-inf', '-infinity', '-∞']:
                            point_str = '-oo'
                        
                        try:
                            # Assume variable is x if not specified
                            func = sympify(func_str, locals=self.symbols_dict)
                            var = sp_symbols('x')
                            point = sympify(point_str, locals=self.symbols_dict)
                            
                            result = limit(func, var, point)
                            return result, None
                        except Exception as e:
                            return None, f"Error parsing limit expression: {str(e)}"
                
                return None, "Invalid limit format"
                
        except Exception as e:
            return None, f"Error solving limit problem: {str(e)}"
    
    def solve_equation(self, equation_text: str) -> Tuple[Any, str]:
        """
        Main equation solving method that routes to appropriate solver.
        
        Args:
            equation_text: Raw equation text
            
        Returns:
            Tuple of (result, error_message)
        """
        if not equation_text or equation_text.isspace():
            return None, "No equation detected"
        
        # Clean the equation text using the function from utils.py
        from utils import clean_equation_text
        cleaned_text = clean_equation_text(equation_text)
        
        try:
            # Determine equation type and route to appropriate solver
            if ';' in cleaned_text and cleaned_text.count('=') > 1:
                # System of equations
                return self.solve_system_of_equations(cleaned_text)
            elif 'diff(' in cleaned_text or 'd/dx' in cleaned_text or 'd/dt' in cleaned_text:
                # Differential equation
                return self.solve_differential_equation(cleaned_text)
            elif '[[' in cleaned_text and ']]' in cleaned_text:
                # Matrix equation
                return self.solve_matrix_equation(cleaned_text)
            elif any(word in cleaned_text.lower() for word in ['minimize', 'maximize', 'min', 'max']):
                # Optimization problem
                return self.solve_optimization_problem(cleaned_text)
            elif any(word in cleaned_text.lower() for word in ['mean', 'variance', 'std', 'average']):
                # Statistical equation
                return self.solve_statistical_equation(cleaned_text)
            elif any(word in cleaned_text.lower() for word in ['integrate', 'integral']):
                # Integration problem - try symbolic integration first
                result, error = self.solve_integration_problem(cleaned_text)
                if result is not None:
                    return result, error
                # If symbolic integration fails, try numerical integration
                try:
                    from scientific_solver import solve_scientific_equation
                    return solve_scientific_equation(cleaned_text, 'integration')
                except (ImportError, Exception):
                    return result, error
            elif any(word in cleaned_text.lower() for word in ['limit', 'lim']) or '->' in cleaned_text:
                # Limit problem
                return self.solve_limit_problem(cleaned_text)
            elif '=' in cleaned_text:
                # Regular algebraic equation
                return self.solve_algebraic_equation(cleaned_text)
            else:
                # Mathematical expression evaluation
                result = sympify(cleaned_text, locals={**self.symbols_dict, **self.functions_dict})
                return result, None
                
        except Exception as e:
            # Last resort: try scientific solver for any remaining cases
            try:
                from scientific_solver import solve_scientific_equation
                result, error = solve_scientific_equation(cleaned_text, 'auto')
                if result is not None:
                    return result, error
            except (ImportError, Exception):
                pass
            
            return None, f"Unable to process equation: {str(e)}"
    
    def _process_derivative_notation(self, text: str, var: str) -> str:
        """
        Process derivative notation like d/dx.
        """
        notation = f'd/d{var}'
        if notation in text:
            # Simple replacement - this could be enhanced further
            text = text.replace(notation, f'diff(,{var})')
            # Additional processing would be needed for complex cases
        return text
    
    def _parse_matrix(self, text: str) -> Matrix:
        """
        Parse matrix from string representation.
        """
        try:
            # Remove spaces and parse
            text = text.strip().replace(' ', '')
            
            # Handle different matrix formats
            if '[' in text and ']' in text:
                # Standard format [[1,2],[3,4]]
                try:
                    matrix_data = eval(text)
                    return Matrix(matrix_data)
                except:
                    # If eval fails, try manual parsing
                    import ast
                    matrix_data = ast.literal_eval(text)
                    return Matrix(matrix_data)
            else:
                # Try to parse as a simple 2D array
                # Handle format like "1,2;3,4" or "1 2;3 4"
                text = text.replace(',', ' ')
                rows = text.split(';')
                matrix_data = []
                for row in rows:
                    if row.strip():  # Skip empty rows
                        row_data = [float(x) for x in row.strip().split()]
                        matrix_data.append(row_data)
                return Matrix(matrix_data)
                
        except Exception as e:
            print(f"Matrix parsing error: {e}")
            return None
    
    def _solve_optimization(self, objective: str, opt_type: str) -> Any:
        """
        Solve optimization problems.
        """
        # Simplified optimization - would need more sophisticated handling
        try:
            expr = sympify(objective, locals=self.symbols_dict)
            variables = list(expr.free_symbols)
            if variables:
                # Find critical points
                derivatives = [diff(expr, var) for var in variables]
                critical_points = solve(derivatives, variables)
                return critical_points
            return expr
        except:
            return f"Cannot solve {opt_type} problem"
    
    def _solve_statistical_formula(self, text: str, stat_type: str) -> Any:
        """
        Solve statistical formulas.
        """
        # Simplified statistical handling
        try:
            if stat_type == 'mean':
                return "Mean formula implementation needed"
            elif stat_type == 'variance':
                return "Variance formula implementation needed"
            elif stat_type == 'std':
                return "Standard deviation formula implementation needed"
            else:
                return sympify(text, locals=self.symbols_dict)
        except:
            return f"Cannot solve {stat_type} formula"

# Create a global instance for backward compatibility
enhanced_solver = EnhancedEquationSolver()

def solve_equation_enhanced(equation_text: str) -> Tuple[Any, str]:
    """
    Enhanced equation solving function for backward compatibility.
    
    Args:
        equation_text: Raw equation text
        
    Returns:
        Tuple of (result, error_message)
    """
    logger.debug(f"solve_equation_enhanced called with: {equation_text}")
    return enhanced_solver.solve_equation(equation_text)
