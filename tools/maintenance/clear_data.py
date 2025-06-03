import os
import shutil
import argparse

def clear_data(data_folder, include_logs=False):
    # Iterate through all items in the data folder
    for item in os.listdir(data_folder):
        item_path = os.path.join(data_folder, item)
        # Check if the item is the logs folder and should be skipped
        if item == 'logs' and not include_logs:
            continue
        # If it's a directory, remove it and all its contents
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        # If it's a file, remove it
        elif os.path.isfile(item_path):
            os.remove(item_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clear data folder contents')
    parser.add_argument('--include-logs', action='store_true', 
                       help='Include logs folder in cleanup')
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    data_folder = os.path.join(parent_dir, 'data')
    
    clear_data(data_folder, args.include_logs)
    print(f"Data folder cleared{', including logs' if args.include_logs else ', except for the logs folder'}.")