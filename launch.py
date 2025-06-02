#!/usr/bin/env python3
"""
Hephia Launcher - one command to rule them all
Handles venv, dependencies, and process management automatically
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import psutil
import time
import json

def get_venv_paths():
    """Get platform-specific venv paths"""
    if platform.system() == "Windows":
        return {
            'python': Path('.venv/Scripts/python.exe'),
            'activate': Path('.venv/Scripts/activate.bat')
        }
    else:
        return {
            'python': Path('.venv/bin/python'),
            'activate': Path('.venv/bin/activate')
        }
    
def load_services_config():
    """Load services configuration"""
    config_path = Path('services.json')
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except json.JSONDecodeError:
            print("error: invalid services.json")
            return {}
    return {}

def get_service_processes():
    """Get running processes for each service"""
    services_config = load_services_config()
    services = services_config.get('services', {})
    
    service_processes = {}
    for service_id, service in services.items():
        script = service['script']
        service_processes[service_id] = []
        
        for proc in find_hephia_processes():
            try:
                cmdline = ' '.join(proc.cmdline())
                if script in cmdline:
                    service_processes[service_id].append(proc)
            except:
                pass
    
    return service_processes

def run_service_monitored(service_id):
    """Run a specific service with monitoring"""
    services_config = load_services_config()
    services = services_config.get('services', {})
    
    if service_id not in services:
        print(f"unknown service: {service_id}")
        return
    
    service = services[service_id]
    
    # Check dependencies
    deps = service.get('depends_on', [])
    service_procs = get_service_processes()
    
    missing_deps = []
    for dep in deps:
        if not service_procs.get(dep):
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"{service['name']} requires: {', '.join(missing_deps)}")
        choice = input("start missing dependencies? (y/n): ").lower().strip()
        if choice == 'y':
            for dep in missing_deps:
                run_service_background(dep)
                time.sleep(2)  # give it time to start
        else:
            print("cannot start service without dependencies")
            return
    
    # Check if service is already running
    if service_procs.get(service_id):
        print(f"{service['name']} already running!")
        choice = input("restart it? (y/n): ").lower().strip()
        if choice == 'y':
            # Kill existing instances
            for proc in service_procs[service_id]:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    try:
                        proc.kill()
                    except:
                        pass
            time.sleep(1)
        else:
            return
    
    # Run the service
    venv_python = get_venv_paths()['python']
    script_path = Path(service['script'])
    
    print(f"starting {service['name']}...")
    
    try:
        proc = subprocess.Popen([str(venv_python), str(script_path)])
        print(f"{service['name']} running (PID: {proc.pid})")
        print("press ctrl+c to stop service")
        
        proc.wait()
        print(f"{service['name']} exited")
        
    except KeyboardInterrupt:
        print(f"\nstopping {service['name']}...")
        try:
            proc.terminate()
            proc.wait(timeout=5)
            print(f"{service['name']} stopped gracefully")
        except subprocess.TimeoutExpired:
            print(f"{service['name']} didn't respond, force killing...")
            proc.kill()
            print(f"{service['name']} killed")
    
def run_service_background(service_id):
    """Start a service in background"""
    services_config = load_services_config()
    services = services_config.get('services', {})
    
    if service_id not in services:
        print(f"unknown service: {service_id}")
        return False
    
    service = services[service_id]
    venv_python = get_venv_paths()['python']
    script_path = Path(service['script'])
    
    print(f"starting {service['name']} in background...")
    
    if platform.system() == "Windows":
        proc = subprocess.Popen([str(venv_python), str(script_path)], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        proc = subprocess.Popen([str(venv_python), str(script_path)], 
                               start_new_session=True)
    
    print(f"{service['name']} started (PID: {proc.pid})")
    return True

def show_service_status():
    """Show status of all services"""
    services_config = load_services_config()
    services = services_config.get('services', {})
    
    if not services:
        print("no services configured")
        return
    
    service_procs = get_service_processes()
    
    print("service status:")
    for service_id, service in services.items():
        procs = service_procs.get(service_id, [])
        if procs:
            pids = [str(proc.pid) for proc in procs]
            print(f"  ✅ {service['name']} (PIDs: {', '.join(pids)})")
        else:
            print(f"  ❌ {service['name']} (not running)")

def manage_services():
    """Service management submenu"""
    services_config = load_services_config()
    services = services_config.get('services', {})
    
    if not services:
        print("no services configured")
        return
    
    while True:
        print("\n=== service management ===")
        service_procs = get_service_processes()
        
        choices = {}
        i = 1
        for service_id, service in services.items():
            is_running = bool(service_procs.get(service_id))
            status = "🟢 running" if is_running else "🔴 stopped"
            print(f"{i}. {service['name']} ({status})")
            choices[str(i)] = service_id
            i += 1
        
        print("a. start all services")
        print("s. show detailed status")
        print("b. back to main menu")
        
        choice = input("\nmanage which service? ").strip().lower()
        if choice == 'b':
            return
        elif choice == 'a':
            start_all_services()
        elif choice == 's':
            show_service_status()
            input("press enter to continue...")
        elif choice in choices:
            manage_single_service(choices[choice], services[choices[choice]])

def manage_single_service(service_id, service_config):
    """Manage a single service"""
    service_procs = get_service_processes()
    is_running = bool(service_procs.get(service_id))
    
    print(f"\n=== {service_config['name']} ===")
    print(f"status: {'running' if is_running else 'stopped'}")
    print(f"script: {service_config['script']}")
    
    if is_running:
        print("1. stop service")
        print("2. restart service")
        print("3. view in foreground (monitor)")
    else:
        print("1. start service (background)")
        print("2. start service (foreground)")
    
    print("b. back")
    
    choice = input("what do you want to do? ").strip()
    
    if choice == 'b':
        return
    elif choice == '1':
        if is_running:
            # Stop service
            for proc in service_procs[service_id]:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    try:
                        proc.kill()
                    except:
                        pass
            print(f"{service_config['name']} stopped")
        else:
            # Start in background
            run_service_background(service_id)
    elif choice == '2':
        if is_running:
            # Restart
            # Stop first
            for proc in service_procs[service_id]:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    try:
                        proc.kill()
                    except:
                        pass
            time.sleep(1)
            # Start in background
            run_service_background(service_id)
        else:
            # Start in foreground
            run_service_monitored(service_id)
    elif choice == '3' and is_running:
        print("note: this will show output but service will keep running in background")
        input("press enter when ready, then ctrl+c to return to menu...")
        # This is tricky - we can't easily "attach" to running process
        # Instead, suggest using logs or restarting in foreground
        print("tip: to monitor a running service, check logs or restart in foreground mode")

def start_all_services():
    """Start all services in dependency order"""
    services_config = load_services_config()
    services = services_config.get('services', {})
    
    # Sort by dependencies
    sorted_services = []
    remaining = list(services.keys())
    
    while remaining:
        for service_id in remaining[:]:
            service = services[service_id]
            deps = service.get('depends_on', [])
            if all(dep in sorted_services for dep in deps):
                sorted_services.append(service_id)
                remaining.remove(service_id)
    
    for service_id in sorted_services:
        run_service_background(service_id)
        time.sleep(1)  # brief pause between services

def load_tools_manifest():
    """Load tools manifest"""
    manifest_path = Path('tools/manifest.json')
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            print("error: invalid tools manifest")
            return {}
    return {}

def show_tools_menu():
    """Enhanced tools menu with category browsing"""
    manifest = load_tools_manifest()
    if not manifest:
        print("no tools manifest found - check tools/manifest.json")
        return
        
    while True:
        print("\n=== tools ===")
        categories = manifest.get('categories', {})
        
        if not categories:
            print("no tool categories found")
            return
        
        cat_choices = {}
        i = 1
        for cat_id, category in categories.items():
            tool_count = len(category.get('tools', {}))
            print(f"{i}. {category['name']} ({tool_count} tools)")
            cat_choices[str(i)] = cat_id
            i += 1
        
        print("b. back to main menu")
        
        choice = input("\nwhich category? ").strip().lower()
        if choice == 'b':
            return
            
        if choice in cat_choices:
            show_category_tools(cat_choices[choice], categories[cat_choices[choice]])

def show_category_tools(cat_id, category):
    """Show tools in a specific category"""
    while True:
        print(f"\n=== {category['name']} ===")
        tools = category.get('tools', {})
        
        if not tools:
            print("no tools in this category")
            input("press enter to go back...")
            return
        
        tool_choices = {}
        i = 1
        for tool_id, tool in tools.items():
            tool_type = tool.get('type', 'simple')
            type_indicator = {"simple": "🔧", "args": "⚙️", "cli": "💻"}.get(tool_type, "❓")
            print(f"{i}. {type_indicator} {tool_id}: {tool['description']}")
            tool_choices[str(i)] = tool_id
            i += 1
            
        print("b. back to categories")
        
        choice = input("\nwhich tool? ").strip().lower()
        if choice == 'b':
            return
            
        if choice in tool_choices:
            tool_id = tool_choices[choice]
            run_tool_enhanced(tool_id, tools[tool_id])

def run_tool_enhanced(tool_id, tool_config):
    """Run tool based on its type"""
    ensure_venv()  # make sure venv is ready
    
    tool_type = tool_config.get('type', 'simple')
    venv_python = get_venv_paths()['python']
    script_path = Path('tools') / tool_config['file']
    
    if not script_path.exists():
        print(f"error: tool script not found: {script_path}")
        input("press enter to continue...")
        return
    
    # Handle confirmation
    if tool_config.get('confirm'):
        print(f"\n⚠️  WARNING: this will {tool_config['description']}")
        if input("are you sure? (y/n): ").lower() != 'y':
            return
    
    try:
        if tool_type == 'simple':
            run_simple_tool(tool_config, venv_python, script_path)
        elif tool_type == 'args':
            run_args_tool(tool_config, venv_python, script_path)
        elif tool_type == 'cli':
            run_cli_tool(tool_id, tool_config, venv_python, script_path)
        else:
            print(f"unknown tool type: {tool_type}")
            
    except KeyboardInterrupt:
        print("\ntool interrupted by user")
    except Exception as e:
        print(f"error running tool: {e}")
    
    input("\npress enter to continue...")

def run_simple_tool(tool_config, venv_python, script_path):
    """Run a simple script with no arguments"""
    print(f"running {tool_config['description']}...")
    subprocess.run([str(venv_python), str(script_path)])

def run_args_tool(tool_config, venv_python, script_path):
    """Run script with arguments"""
    args = build_args_for_tool(tool_config)
    if args is None:  # user cancelled or error
        return
        
    cmd = [str(venv_python), str(script_path)] + args
    print(f"running: {script_path.name} {' '.join(args)}")
    subprocess.run(cmd)

def build_args_for_tool(tool_config):
    """Build arguments for tools that take args"""
    args = []
    
    for arg_config in tool_config.get('args', []):
        arg_name = arg_config['name']
        arg_desc = arg_config['description']
        
        if arg_config.get('type') == 'flag':
            # Yes/no question for flags
            prompt = f"use {arg_name}? ({arg_desc}) (y/n): "
            if input(prompt).lower().strip() == 'y':
                args.append(arg_name)
                
        elif arg_config.get('required'):
            # Required argument
            value = input(f"{arg_name} ({arg_desc}): ").strip()
            if not value:
                print("required argument cannot be empty!")
                return None
            args.append(value)
            
        else:
            # Optional argument
            value = input(f"{arg_name} ({arg_desc}) [optional]: ").strip()
            if value:
                args.extend([arg_name, value])
                
    return args

def run_cli_tool(tool_id, tool_config, venv_python, script_path):
    """Handle complex CLI tools with subcommands"""
    subcommands = tool_config.get('subcommands', {})
    
    if not subcommands:
        print("no subcommands defined for this CLI tool")
        return
    
    while True:
        print(f"\n=== {tool_config['description']} ===")
        
        sub_choices = {}
        i = 1
        for sub_id, sub_config in subcommands.items():
            print(f"{i}. {sub_id}: {sub_config['description']}")
            sub_choices[str(i)] = sub_id
            i += 1
            
        print("b. back")
        
        choice = input("\nwhich command? ").strip().lower()
        if choice == 'b':
            return
            
        if choice in sub_choices:
            sub_id = sub_choices[choice]
            run_cli_subcommand(sub_id, subcommands[sub_id], venv_python, script_path)

def run_cli_subcommand(sub_id, sub_config, venv_python, script_path):
    """Run a specific CLI subcommand"""
    cmd = [str(venv_python), str(script_path), sub_id]
    
    # Build arguments for this subcommand
    for arg_config in sub_config.get('args', []):
        arg_name = arg_config['name']
        arg_desc = arg_config['description']
        
        if arg_config.get('required'):
            value = input(f"{arg_name} ({arg_desc}): ").strip()
            if not value:
                print("required argument cannot be empty!")
                return
            cmd.append(value)
            
        elif arg_config.get('multiple'):
            # Handle multiple values like --param key=value --param key2=value2
            print(f"{arg_name} ({arg_desc}) - enter multiple values:")
            while True:
                value = input(f"  {arg_name} [enter empty to stop]: ").strip()
                if not value:
                    break
                cmd.extend([arg_name, value])
                
        else:
            # Optional single argument
            value = input(f"{arg_name} ({arg_desc}) [optional]: ").strip()
            if value:
                cmd.extend([arg_name, value])
    
    print(f"running: {script_path.name} {' '.join(cmd[2:])}")
    subprocess.run(cmd)

def find_hephia_processes():
    """Find any running hephia processes"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if any(x in cmdline for x in ['main.py', 'client.tui', 'client.config', 'hephia']):
                # exclude this launcher process
                if 'launch.py' not in cmdline:
                    processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return processes

def kill_existing_processes():
    """Kill any existing hephia processes"""
    procs = find_hephia_processes()
    if not procs:
        print("no existing hephia processes found")
        return
        
    print(f"found {len(procs)} running hephia processes:")
    for proc in procs:
        try:
            cmdline = ' '.join(proc.cmdline())
            print(f"  PID {proc.pid}: {cmdline}")
        except:
            print(f"  PID {proc.pid}: <access denied>")
    
    choice = input("\nkill them? (y/n): ").lower().strip()
    if choice == 'y':
        killed = 0
        for proc in procs:
            try:
                proc.terminate()
                proc.wait(timeout=3)
                killed += 1
            except psutil.TimeoutExpired:
                try:
                    proc.kill()  # force kill
                    killed += 1
                except:
                    pass
            except:
                pass
        print(f"killed {killed}/{len(procs)} processes")
    else:
        print("keeping existing processes")

def ensure_venv():
    """Create venv if it doesn't exist, check if it does"""
    venv_python = get_venv_paths()['python']
    
    if not venv_python.exists():
        print("no venv found, running install.py...")
        try:
            subprocess.run([sys.executable, 'install.py'], check=True)
            print("setup complete!")
        except subprocess.CalledProcessError:
            print("setup failed! check install.py output")
            sys.exit(1)
        return
    
    print(f"using venv: {venv_python}")

def run_server():
    """Start server with proper monitoring"""
    # check for conflicts first
    server_procs = []
    for proc in find_hephia_processes():
        try:
            cmdline = ' '.join(proc.cmdline())
            if 'main.py' in cmdline:
                server_procs.append(proc)
        except:
            pass
    
    if server_procs:
        print("server already running!")
        choice = input("kill existing server and start new one? (y/n): ").lower().strip()
        if choice != 'y':
            return
        
        for proc in server_procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass
        time.sleep(1)  # give it a moment
    
    venv_python = get_venv_paths()['python']
    print("starting server...")
    
    try:
        proc = subprocess.Popen([str(venv_python), 'main.py'])
        print(f"server running (PID: {proc.pid})")
        print("server output will appear above")
        print("press ctrl+c to stop server and return to launcher")
        
        proc.wait()  # wait for server to exit naturally
        print("server exited")
        
    except KeyboardInterrupt:
        print("\nstopping server...")
        try:
            proc.terminate()
            proc.wait(timeout=5)
            print("server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("server didn't respond, force killing...")
            proc.kill()
            print("server killed")
        except:
            print("server cleanup completed")

def run_monitor():
    """Start monitor TUI"""
    venv_python = get_venv_paths()['python']
    print("starting monitor tui...")
    print("(this will take over the terminal)")
    time.sleep(1)
    
    try:
        subprocess.run([str(venv_python), '-m', 'client.tui'])
    except KeyboardInterrupt:
        print("\nmonitor exited")

def run_config():
    """Start config TUI"""
    venv_python = get_venv_paths()['python']
    print("starting config tool...")
    print("(this will take over the terminal)")
    time.sleep(1)
    
    try:
        subprocess.run([str(venv_python), '-m', 'client.config'])
    except KeyboardInterrupt:
        print("\nconfig tool exited")

def sync_dependencies():
    """Sync dependencies using install.py"""
    print("syncing dependencies...")
    try:
        subprocess.run([sys.executable, 'install.py'])
        print("dependencies synced!")
    except subprocess.CalledProcessError:
        print("dependency sync failed")

def show_status():
    """Show current hephia processes"""
    procs = find_hephia_processes()
    if not procs:
        print("no hephia processes running")
        return
    
    print(f"running processes ({len(procs)}):")
    for proc in procs:
        try:
            cmdline = ' '.join(proc.cmdline())
            if 'main.py' in cmdline:
                print(f"  server (PID: {proc.pid})")
            elif 'client.tui' in cmdline:
                print(f"  monitor (PID: {proc.pid})")
            elif 'client.config' in cmdline:
                print(f"  config (PID: {proc.pid})")
            else:
                print(f"  unknown (PID: {proc.pid}): {cmdline}")
        except:
            print(f"  process (PID: {proc.pid})")

def main():
    print("╔══════════════════════════════╗")
    print("║       hephia launcher        ║")  
    print("╚══════════════════════════════╝")
    
    while True:
        print("\nwhat do you want to do?")
        print("1. start main server")
        print("2. start discord bot")
        print("3. start both services")
        print("4. start monitor") 
        print("5. start config tool")
        print("6. tools & utilities")
        print("7. manage services")
        print("8. show running processes")
        print("9. kill all hephia processes")
        print("10. sync dependencies")
        print("11. quit")
        
        try:
            choice = input("\npick one (1-11): ").strip()
            
            if choice == '1':
                ensure_venv()
                run_service_monitored('hephia-main')
            elif choice == '2':
                ensure_venv()
                run_service_monitored('hephia-discord')
            elif choice == '3':
                ensure_venv()
                start_all_services()
                print("all services started in background")
            elif choice == '4':
                ensure_venv() 
                run_monitor()
            elif choice == '5':
                ensure_venv()
                run_config()
            elif choice == '6':
                show_tools_menu()
            elif choice == '7':
                manage_services()
            elif choice == '8':
                show_service_status()
                input("press enter to continue...")
            elif choice == '9':
                kill_existing_processes()
            elif choice == '10':
                sync_dependencies()
            elif choice == '11':
                print("bye!")
                break
            else:
                print("invalid choice, try again")
                
        except KeyboardInterrupt:
            print("\n\nbye!")
            break
        except Exception as e:
            print(f"error: {e}")

if __name__ == "__main__":
    main()