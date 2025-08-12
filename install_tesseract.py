import os
import sys
import subprocess
import urllib.request
import tempfile
import winreg
import ctypes

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def download_file(url, save_path):
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def run_installer(installer_path):
    print(f"Running installer: {installer_path}")
    try:
        # Run the installer silently with default options
        result = subprocess.run(
            [installer_path, '/S'], 
            check=True
        )
        print("Installation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running installer: {e}")
        return False

def add_to_path(directory):
    try:
        # Open the registry key for the system PATH
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 0, winreg.KEY_ALL_ACCESS)
        
        # Get the current PATH value
        path, _ = winreg.QueryValueEx(key, 'Path')
        
        # Check if the directory is already in the PATH
        if directory.lower() not in path.lower():
            # Add the directory to the PATH
            new_path = path + ';' + directory
            winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)
            print(f"Added {directory} to system PATH")
        else:
            print(f"{directory} is already in system PATH")
            
        winreg.CloseKey(key)
        return True
    except Exception as e:
        print(f"Error adding to PATH: {e}")
        return False

def main():
    if not is_admin():
        print("This script requires administrator privileges to install Tesseract and modify the PATH.")
        print("Please run this script as administrator.")
        return False
    
    # Tesseract installer URL (64-bit version)
    tesseract_url = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe"
    
    # Download location
    temp_dir = tempfile.gettempdir()
    installer_path = os.path.join(temp_dir, "tesseract-installer.exe")
    
    # Download the installer
    if not download_file(tesseract_url, installer_path):
        return False
    
    # Run the installer
    if not run_installer(installer_path):
        return False
    
    # Add Tesseract to PATH
    tesseract_dir = r"C:\Program Files\Tesseract-OCR"
    if not add_to_path(tesseract_dir):
        return False
    
    print("\nTesseract OCR has been successfully installed!")
    print(f"Installation directory: {tesseract_dir}")
    print("\nYou may need to restart your application or computer for the PATH changes to take effect.")
    return True

if __name__ == "__main__":
    print("=== Tesseract OCR Installer ===\n")
    success = main()
    
    if success:
        print("\nInstallation completed successfully!")
    else:
        print("\nInstallation failed. Please check the error messages above.")
    
    input("\nPress Enter to exit...")