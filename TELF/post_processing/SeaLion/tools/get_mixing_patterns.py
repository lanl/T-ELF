import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

def mixing_patterns(S:np.ndarray, save_path:str):
    plt.figure(dpi=300)
    sns.heatmap(S, 
                square=True, norm=LogNorm(), # Uncomment this line if want to do Log scale
                cmap="YlOrRd",
                vmin=0,
                linewidths=0.1,
                linecolor="black",
                annot=True
                )

    plt.title("S Mixing Patterns")
    plt.xlabel("Clusters")
    plt.ylabel("Patterns")

    plt.tight_layout()
    plt.savefig(f'{save_path}S_mixing_patterns.png')
    plt.close()