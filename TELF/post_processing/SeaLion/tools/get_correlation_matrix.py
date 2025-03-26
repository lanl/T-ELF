import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

def corr_matrix(A:np.ndarray, name:str, save_path:str):
        
    A_corr = np.corrcoef(A, rowvar=False)
    plt.figure(dpi=300)
    mask = np.triu(np.ones_like(A_corr, dtype=bool))
    sns.heatmap(A_corr, 
                square=True, 
                norm=LogNorm(), # Uncomment this line if want to do Log scale
                cmap="YlOrRd",
                #linewidths=0.1,
                robust=True,
                #linecolor="black",
                annot=True,
                )

    plt.xlabel("Patterns")
    plt.ylabel("Patterns")

    plt.title(f'{name} Pattern Correlations')
    plt.tight_layout()
    plt.savefig(f'{save_path}/{name}_correlation_matrix_heatmap.png')
    plt.close()
    
    pd.DataFrame(A_corr).to_csv(f'{save_path}/{name}_correlation_matrix_table.csv', index=False)
