@echo off
echo Starting Enhanced Handwritten Equation Solver...
echo.
echo ✅ Features:
echo   - Upload images of handwritten equations
echo   - Draw equations on canvas
echo   - Enhanced OCR for chalk-on-blackboard style text
echo   - Multiple preprocessing approaches for better recognition
echo   - Automatic equation solving with SymPy
echo.
echo The application will be available at: http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
echo.

REM Activate virtual environment and run the Flask app
call venv\Scripts\activate.bat && python app.py

pause
