from pathlib import Path
import pathlib
import numpy as np
import os
import pandas as pd
from ....helpers.stats import top_words

def words_probabilities(W:np.ndarray, save_path:str, rows_name:str, rows:list, num_top_words:int):
    words, probabilities = top_words(W, rows, num_top_words)
    
    words_df = pd.DataFrame(words)
    words_df.to_csv(f'{save_path}/top_{rows_name}.csv', index=False)
    
    probabilities_df = pd.DataFrame(probabilities)
    probabilities_df.to_csv(f'{save_path}/top_{rows_name}_probabilities.csv', index=False)

    for k in range(W.shape[1]):
        k_words_df = pd.DataFrame.from_dict({rows_name:words_df[k], "probabilities":probabilities_df[k]})
        k_words_df.to_csv(f'{os.path.join(save_path, str(k))}/top_{rows_name}_in_cluster_{k}.csv', index=False)
         
    return words, probabilities