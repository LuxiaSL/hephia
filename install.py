#!/usr/bin/env python
"""Environment setup script for Hephia using uv package manager"""

import subprocess
import sys
import os
import platform
import argparse
from pathlib import Path
import json
import getpass
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

# Add these imports
import argparse
import json

def load_services_config():
    """Load services configuration"""
    config_path = Path('services.json')
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except json.JSONDecodeError:
            print_status("Invalid services.json file", "error")
            return {}
    return {}

def list_available_services():
    """List all available services"""
    services_config = load_services_config()
    services = services_config.get('services', {})
    
    if not services:
        print_status("No services defined", "warning")
        return
    
    print("\nAvailable services:")
    for service_id, service in services.items():
        deps = service.get('depends_on', [])
        deps_str = f" (depends on: {', '.join(deps)})" if deps else ""
        print(f"  • {service_id}: {service['description']}{deps_str}")

def install_single_service(service_id, service_config, system):
    """Install a single service for the given system"""
    service_name = service_config['name']
    script_path = service_config['script']
    
    print_status(f"Installing {service_name}...", "info")
    
    # Get absolute paths
    project_dir = Path.cwd().absolute()
    if system == 'windows':
        python_path = project_dir / '.venv/Scripts/python.exe'
    else:
        python_path = project_dir / '.venv/bin/python'
    
    full_script_path = project_dir / script_path
    if not full_script_path.exists():
        print_status(f"Service script not found: {full_script_path}", "error")
        return False
    
    try:
        if system == 'linux':
            install_linux_service(service_id, service_config, project_dir, python_path)
        elif system == 'windows':
            install_windows_service(service_id, service_config, project_dir, python_path)
        elif system == 'darwin':
            install_macos_service(service_id, service_config, project_dir, python_path)
        
        print_status(f"{service_name} service installed", "success")
        return True
        
    except Exception as e:
        print_status(f"Failed to install {service_name}: {e}", "error")
        return False

def install_linux_service(service_id, service_config, project_dir, python_path):
    """Install systemd user service"""
    systemd_dir = Path.home() / '.config/systemd/user'
    systemd_dir.mkdir(parents=True, exist_ok=True)
    
    # Create service file
    service_content = f"""[Unit]
Description={service_config['name']}
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory={project_dir}
ExecStart={python_path} {service_config['script']}
Restart={'always' if service_config.get('auto_restart') else 'no'}
RestartSec=10
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
"""
    
    service_path = systemd_dir / f'{service_id}.service'
    service_path.write_text(service_content)
    
    # Reload systemd and enable service
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', service_id], check=True)

def install_windows_service(service_id, service_config, project_dir, python_path):
    """Install Windows scheduled task"""
    import getpass
    
    task_xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2">
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{getpass.getuser()}</UserId>
    </LogonTrigger>
  </Triggers>
  <Actions>
    <Exec>
      <Command>{python_path}</Command>
      <Arguments>{service_config['script']}</Arguments>
      <WorkingDirectory>{project_dir}</WorkingDirectory>
    </Exec>
  </Actions>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>false</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Principals>
    <Principal>
      <UserId>{getpass.getuser()}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
</Task>"""
    
    temp_xml = project_dir / f'{service_id}_task.xml'
    temp_xml.write_text(task_xml, encoding='utf-16')
    
    try:
        subprocess.run(['schtasks', '/create', '/tn', service_id, '/xml', str(temp_xml)], check=True)
    finally:
        if temp_xml.exists():
            temp_xml.unlink()

def install_macos_service(service_id, service_config, project_dir, python_path):
    """Install macOS LaunchAgent"""
    agents_dir = Path.home() / 'Library/LaunchAgents'
    agents_dir.mkdir(parents=True, exist_ok=True)
    
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hephia.{service_id}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{service_config['script']}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{project_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>"""
    
    plist_path = agents_dir / f'com.hephia.{service_id}.plist'
    plist_path.write_text(plist_content)
    
    subprocess.run(['launchctl', 'load', str(plist_path)], check=True)

def install_services(service_names=None):
    """Install one or more services"""
    services_config = load_services_config()
    services = services_config.get('services', {})
    
    if not services:
        print_status("No services configuration found", "error")
        return False
    
    if service_names is None:
        # install all services
        service_names = list(services.keys())
    elif isinstance(service_names, str):
        service_names = [s.strip() for s in service_names.split(',')]
    
    system = platform.system().lower()
    if system not in ['linux', 'windows', 'darwin']:
        print_status("Service installation not supported on this platform", "error")
        return False
    
    success_count = 0
    
    # Sort services by dependencies
    sorted_services = []
    remaining = service_names.copy()
    
    while remaining:
        for service_name in remaining[:]:
            service = services[service_name]
            deps = service.get('depends_on', [])
            if all(dep in sorted_services or dep not in service_names for dep in deps):
                sorted_services.append(service_name)
                remaining.remove(service_name)
        
        if len(remaining) == len(sorted_services):
            print_status("Circular dependency detected in services", "error")
            break
    
    for service_name in sorted_services:
        if service_name in services:
            if install_single_service(service_name, services[service_name], system):
                success_count += 1
        else:
            print_status(f"Unknown service: {service_name}", "error")
    
    if success_count > 0:
        # Create log directory
        log_dir = Path.home() / '.hephia/logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        create_service_management_scripts(system)
        print_status(f"Installed {success_count} services successfully", "success")
        return True
    
    return False

def create_service_management_scripts(system):
    """Create service management scripts"""
    services_config = load_services_config()
    services = services_config.get('services', {})
    
    if system == 'linux':
        script_content = """#!/bin/bash
# Hephia service management script

SERVICE_NAME="$2"
SERVICES=(""" + " ".join(f'"{s}"' for s in services.keys()) + """)

if [ -z "$SERVICE_NAME" ]; then
    echo "Available services: ${SERVICES[@]}"
    echo "Usage: $0 {start|stop|status|restart} <service_name>"
    echo "       $0 {start-all|stop-all|restart-all}"
    exit 1
fi

case "$1" in
    start)
        systemctl --user start "$SERVICE_NAME"
        echo "$SERVICE_NAME started"
        ;;
    stop)
        systemctl --user stop "$SERVICE_NAME"
        echo "$SERVICE_NAME stopped"
        ;;
    status)
        systemctl --user status "$SERVICE_NAME"
        ;;
    restart)
        systemctl --user restart "$SERVICE_NAME"
        echo "$SERVICE_NAME restarted"
        ;;
    start-all)
        for service in "${SERVICES[@]}"; do
            systemctl --user start "$service"
            echo "$service started"
        done
        ;;
    stop-all)
        for service in "${SERVICES[@]}"; do
            systemctl --user stop "$service"
            echo "$service stopped"
        done
        ;;
    restart-all)
        for service in "${SERVICES[@]}"; do
            systemctl --user restart "$service"
            echo "$service restarted"
        done
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart} <service_name>"
        echo "       $0 {start-all|stop-all|restart-all}"
        exit 1
        ;;
esac
"""
        script_path = Path('hephia-services.sh')
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        print_status("Created hephia-services.sh for service management", "success")

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
    parser = argparse.ArgumentParser(description='Hephia environment setup script')
    parser.add_argument('--service', nargs='?', const=True,
                       help='Install as background service (optionally specify service names: main,discord)')
    parser.add_argument('--list-services', action='store_true',
                       help='List available services')
    args = parser.parse_args()

    if args.list_services:
        list_available_services()
        return

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

    if args.service:
        if args.service is True:
            # Install all services
            print_status("Installing all Hephia services...", "info")
            if install_services():
                print_status("Service installation complete!", "success")
            else:
                print_status("Service installation failed", "error")
        else:
            # Install specific services
            service_list = args.service
            print_status(f"Installing services: {service_list}", "info")
            if install_services(service_list):
                print_status("Service installation complete!", "success")
            else:
                print_status("Service installation failed", "error")

    print_status(f"Environment setup complete!", "success")
    print_status(f"Activate your virtual environment with: {activate_cmd}", "success")
    if not args.service:
        print_status("Run: python launch.py", "success")
        print_status("you can also do: python install.py --service (to install as background service)", "success")
    else:
        print_status("Hephia is now installed as a service!", "success")

if __name__ == "__main__":
    main()