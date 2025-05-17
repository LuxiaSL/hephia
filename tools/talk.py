#!/usr/bin/env python3

import requests
import colorama
from colorama import Fore, Style
import os
import json
from datetime import datetime

# Attempt to import readline for better input experience on Unix-like systems
try:
    import readline
except ImportError:
    pass  # Readline not available (e.g., on Windows without pyreadline)

colorama.init(autoreset=True)

CHAT_HISTORY_DIR = "chats"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

def ensure_chat_dir():
    """Ensures the chat history directory exists."""
    if not os.path.exists(CHAT_HISTORY_DIR):
        try:
            os.makedirs(CHAT_HISTORY_DIR)
        except OSError as e:
            print(Fore.RED + f"Error creating chat history directory '{CHAT_HISTORY_DIR}': {e}")
            return False
    return True

def save_conversation(conversation, server_url):
    """Saves the conversation to a timestamped JSON file."""
    if not ensure_chat_dir() or not conversation:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize server_url to create a valid filename component
    server_name = server_url.split("//")[-1].replace(":", "_").replace("/", "_")
    filename = os.path.join(CHAT_HISTORY_DIR, f"chat_{server_name}_{timestamp}.json")

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=4)
        print(Fore.YELLOW + f"Conversation saved to {filename}")
    except IOError as e:
        print(Fore.RED + f"Error saving conversation: {e}")
    except Exception as e:
        print(Fore.RED + f"An unexpected error occurred while saving: {e}")


def get_user_input():
    """Gets user input, allowing for a simple multiline mode."""
    prompt = Fore.CYAN + "You: " + Style.RESET_ALL
    
    # Check if the user wants to start multiline input
    first_line = input(prompt)
    if first_line.strip().lower() == "/multiline":
        print(Fore.YELLOW + "Multiline mode enabled. Type an empty line to send, or /cancel to abort.")
        lines = []
        while True:
            line = input(Fore.CYAN + "... " + Style.RESET_ALL)
            if line == "":
                break
            if line.strip().lower() == "/cancel":
                print(Fore.YELLOW + "Multiline input cancelled.")
                return "" # Or handle as a cancelled input
            lines.append(line)
        return "\n".join(lines)
    return first_line

def main():
    server_url = "http://localhost:5517/v1/chat/completions"
    conversation = []

    print(Fore.YELLOW + "Enhanced Chat Client" + Style.RESET_ALL)
    print("Type your message. Use '/multiline' for multi-line input (empty line to send).")
    print("Type '/quit' or '/exit' to save and exit.\n")

    while True:
        user_input = get_user_input()
        
        if not user_input.strip() and "/multiline" not in user_input.lower(): # if multiline was cancelled and returned empty
            continue


        if user_input.strip().lower() in ["/quit", "/exit"]:
            print(Fore.YELLOW + "Exiting chat...")
            break

        current_time = datetime.now().strftime(DATETIME_FORMAT)
        conversation.append({"role": "user", "content": user_input, "timestamp": current_time})

        try:
            payload = {"messages": conversation, "stream": False} # Server handles full history
            response = requests.post(server_url, json=payload, timeout=120)
            response.raise_for_status()

            data = response.json()
            assistant_text = data["choices"][0]["message"]["content"]
            assistant_time = datetime.now().strftime(DATETIME_FORMAT)
            
            conversation.append({"role": "assistant", "content": assistant_text, "timestamp": assistant_time})
            
            print(f"{Fore.MAGENTA}[{assistant_time}] " + Fore.GREEN + "Assistant: " + Style.RESET_ALL + assistant_text)

        except requests.RequestException as e:
            print(Fore.RED + f"Error communicating with server: {e}")
        except (KeyError, IndexError) as e:
            print(Fore.RED + f"Unexpected response format from server: {e}\nResponse text: {response.text}")
        except Exception as e:
            print(Fore.RED + f"An unexpected error occurred: {e}")

    if conversation:
        save_conversation(conversation, server_url)
    print(Fore.YELLOW + "Chat client closed.")

if __name__ == "__main__":
    main()