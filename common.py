import os

def make_dirs(path):
    dir_path = os.path.dirname(path)

    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)