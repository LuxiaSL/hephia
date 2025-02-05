#!/usr/bin/env python3

import requests
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

def main():
    server_url = "http://localhost:8000/v1/chat/completions"

    # We'll keep a local conversation array in memory
    # The roles can be: "system", "user", "assistant"
    # (The server likely injects its own system prompt, so you don't *have* to add it here,
    # but you can if you'd like.)
    conversation = []

    print(Fore.YELLOW + "Simple Chat Client" + Style.RESET_ALL)
    print("Type your message, or '/quit' to exit.\n")

    while True:
        # Prompt user
        user_input = input(Fore.CYAN + "You: " + Style.RESET_ALL)
        
        # If the user wants to quit, break out
        if user_input.strip().lower() in ["/quit", "/exit"]:
            print("Exiting...")
            break

        # Append user message to local conversation
        conversation.append({"role": "user", "content": user_input})

        try:
            # Send the entire conversation so far to the server
            payload = {"messages": conversation, "stream": False}
            response = requests.post(server_url, json=payload, timeout=60)
            response.raise_for_status()

            data = response.json()
            # Typically, the returned structure is:
            # {
            #   "id": "...",
            #   "object": "chat.completion",
            #   "created": 1685000000,
            #   "model": "exocortex",
            #   "choices": [
            #       { "index": 0, "message": {"role": "assistant", "content": "..."} }
            #   ]
            # }
            assistant_text = data["choices"][0]["message"]["content"]
            
            # Append assistant message
            conversation.append({"role": "assistant", "content": assistant_text})
            
            # Print response
            print(Fore.GREEN + "Assistant: " + Style.RESET_ALL + assistant_text)

        except requests.RequestException as e:
            print(Fore.RED + "Error communicating with server: " + Style.RESET_ALL + str(e))
        except KeyError:
            print(Fore.RED + "Unexpected response format:" + Style.RESET_ALL, response.text)
        except Exception as e:
            print(Fore.RED + "Error:" + Style.RESET_ALL, str(e))

if __name__ == "__main__":
    main()
