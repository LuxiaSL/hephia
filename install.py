#!/usr/bin/env python
"""Environment setup script for Hephia using uv package manager"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import shutil

def print_status(message, status="info"):
    """Print colored status messages without unicode symbols to avoid encoding issues"""
    if status == "success":
        print(f"SUCCESS: {message}")
    elif status == "warning":
        print(f"WARNING: {message}")
    elif status == "error":
        print(f"ERROR: {message}")
    else:
        print(f"INFO: {message}")

def run_command(cmd, error_msg=None, capture_output=False):
    """Run a command with proper error handling"""
    try:
        print_status(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=capture_output, text=True)
        return True
    except subprocess.CalledProcessError as e:
        if error_msg:
            print_status(f"{error_msg}: {e}", "error")
            if e.stderr:
                print(f"Error details: {e.stderr}")
        else:
            print_status(f"Command failed: {e}", "error")
        return False

def ensure_pip_in_venv(python_path):
    """Make sure pip is properly installed in the virtual environment"""
    # First try to bootstrap pip using get-pip.py
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    get_pip_path = os.path.join(os.getcwd(), "get-pip.py")
    
    print_status("Downloading pip installer...")
    try:
        import urllib.request
        urllib.request.urlretrieve(get_pip_url, get_pip_path)
        print_status("Downloaded pip installer")
    except Exception as e:
        print_status(f"Failed to download pip installer: {e}", "error")
        return False
    
    # Run the pip installer script
    if run_command([python_path, get_pip_path], "Failed to install pip"):
        print_status("Pip installed successfully", "success")
        # Remove the installer script
        os.remove(get_pip_path)
        return True
    
    # If that fails, try using the system pip to install into the venv
    print_status("Trying alternative pip installation method...")
    if run_command([sys.executable, "-m", "pip", "install", "--target", 
                  os.path.dirname(python_path) + "\\Lib\\site-packages", "pip"], 
                 "Failed to install pip with alternative method"):
        print_status("Pip installed successfully using alternative method", "success")
        return True
    
    return False

def main():
    # Check Python version
    major, minor = sys.version_info[:2]
    if not (major == 3 and 9 <= minor <= 12):
        print_status(f"Current Python version is {major}.{minor}, but Hephia requires 3.9-3.12", "warning")
        response = input("Do you want to continue anyway? (y/n): ").lower()
        if response != 'y':
            print_status("Setup aborted", "error")
            return

    # Check if pyproject.toml exists
    if not os.path.exists("pyproject.toml"):
        print_status("pyproject.toml not found! This file is required.", "error")
        return

    # Install uv if needed
    print_status("Checking for uv package manager...")
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print_status("uv is already installed", "success")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status("Installing uv package manager...")
        if not run_command([sys.executable, "-m", "pip", "install", "uv"],
                        "Failed to install uv"):
            return
        print_status("uv installed successfully", "success")

    # Create virtual environment
    venv_dir = Path(".venv")
    if venv_dir.exists():
        print_status("Virtual environment already exists", "warning")
        response = input("Do you want to recreate it? (y/n): ").lower()
        if response == 'y':
            import shutil
            shutil.rmtree(venv_dir)
        else:
            print_status("Using existing virtual environment")
    
    if not venv_dir.exists():
        print_status("Creating virtual environment...")
        # Explicitly use the current Python executable to create the venv
        python_version = f"{major}.{minor}"
        print_status(f"Using Python {python_version} to create virtual environment")
        if not run_command(["uv", "venv", ".venv", "--python", sys.executable],
                        "Failed to create virtual environment"):
            return
        print_status("Virtual environment created", "success")

    # Determine paths for the virtual environment
    if platform.system() == "Windows":
        python_path = os.path.abspath(os.path.join(".venv", "Scripts", "python.exe"))
        pip_path = os.path.abspath(os.path.join(".venv", "Scripts", "pip.exe"))
        activate_cmd = ".venv\\Scripts\\activate"
    else:
        python_path = os.path.abspath(os.path.join(".venv", "bin", "python"))
        pip_path = os.path.abspath(os.path.join(".venv", "bin", "pip"))
        activate_cmd = "source .venv/bin/activate"

    print_status(f"Python path: {python_path}")
    
    # Make sure pip is available in the virtual environment
    if not os.path.exists(pip_path):
        print_status("Pip not found in virtual environment, installing...")
        if not ensure_pip_in_venv(python_path):
            print_status("Failed to install pip, proceeding with uv only", "warning")
    
    # First try installing with uv
    print_status("Installing dependencies with uv...")
    if run_command(["uv", "pip", "install", "--python", python_path, "-e", "."],
                  "Installation with uv failed"):
        print_status("Dependencies installed successfully with uv", "success")
    else:
        # Fall back to pip if uv fails
        print_status("Trying pip installation instead...")
        if os.path.exists(pip_path):
            if not run_command([pip_path, "install", "-e", "."],
                             "Installation with pip failed"):
                print_status("All installation methods failed", "error")
                return
            print_status("Dependencies installed successfully with pip", "success")
        else:
            print_status("Could not find pip, and uv installation failed", "error")
            return

    # Download NLTK data
    print_status("Downloading NLTK data...")
    nltk_script = "import nltk; nltk.download('punkt_tab'); nltk.download('maxent_ne_chunker'); nltk.download('maxent_ne_chunker_tab'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('words')"
    if not run_command([python_path, "-c", nltk_script],
                     "Failed to download NLTK data"):
        return
    print_status("NLTK data downloaded successfully", "success")

    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        print_status("Creating .env file from template...")
        if os.path.exists(".env.example"):
            with open(".env.example", "r") as f_example:
                with open(".env", "w") as f_env:
                    f_env.write(f_example.read())
            print_status(".env file created. Remember to add your API keys!", "warning")
        else:
            print_status("No .env.example file found", "warning")

    print_status(f"Environment setup complete!", "success")
    print_status(f"Activate your virtual environment with: {activate_cmd}", "success")
    print_status("Then run: python main.py", "success")

if __name__ == "__main__":
    main()