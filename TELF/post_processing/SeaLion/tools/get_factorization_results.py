import os
import shutil
from pathlib import Path

def copy_results(factorization_paths: str, save_path:str, factorization_name="factorization"):
    path = os.path.join(save_path, f'{factorization_name}_results')
    if not Path(path).is_dir():
        Path(path).mkdir(parents=True)
    for curr_path in factorization_paths:
        try:
            shutil.copytree(curr_path, os.path.join(path, curr_path.split("/")[-1]))
        except Exception as e:
            print(e)
