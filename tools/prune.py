import requests
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

def prune_conversation():
    """Send a request to prune the conversation history."""
    server_url = "http://localhost:8000/v1/prune_conversation"
    
    try:
        print(Fore.YELLOW + "Pruning conversation history..." + Style.RESET_ALL)
        response = requests.post(server_url, timeout=10)
        response.raise_for_status()
        print(Fore.GREEN + "Successfully pruned conversation history." + Style.RESET_ALL)
        return True
        
    except requests.RequestException as e:
        print(Fore.RED + "Error communicating with server: " + Style.RESET_ALL + str(e))
        return False
    except Exception as e:
        print(Fore.RED + "Error: " + Style.RESET_ALL + str(e))
        return False

if __name__ == "__main__":
    prune_conversation()