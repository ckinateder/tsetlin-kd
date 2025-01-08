import json
import os

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)