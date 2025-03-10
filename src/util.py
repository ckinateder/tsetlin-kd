import json
import os
import shutil
import pickle as pkl
def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_pkl(data, file_path):
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)

def make_dir(dir_path, overwrite=False):
    if os.path.exists(dir_path):
        if overwrite:
            shutil.rmtree(dir_path)
        else:
            print(f"Directory {dir_path} already exists, skipping")
            return
    os.makedirs(dir_path)

def rm_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def load_or_create(file_path:str, create_func, **kwargs):
    if os.path.exists(file_path):
        return load_pkl(file_path)
    else:
        data = create_func(**kwargs)
        save_pkl(data, file_path)
        return data