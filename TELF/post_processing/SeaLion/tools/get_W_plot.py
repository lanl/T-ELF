import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

def W_UMAP(W_sub:np.ndarray, Wsub_name_idx_map:dict, save_path:str, num_top_words:int, args={}):
    fit = umap.UMAP(**args)
    u2 = fit.fit_transform(W_sub.astype("float32"))
    plt.figure(dpi=300)

    group = W_sub.argmax(axis=1)

    for g in np.unique(group):
        ix = np.where(group == g)
        plt.scatter(u2[ix][:,0], u2[ix][:,1], label = f'Cluster={g}', s = 10)

    for sample_idx in range(len(u2)):
        plt.annotate(list(Wsub_name_idx_map.keys())[sample_idx], (u2[sample_idx,0] , u2[sample_idx,1]), fontsize=2, alpha=0.8)

    plt.title(f'UMAP Plot of W Top {num_top_words} Clusters')
    plt.xticks([]) 
    plt.yticks([]) 
    plt.legend(fontsize=6, framealpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_path}/umap_labelled_top-{num_top_words}.png')
    plt.close()

def W_plot(W_sub:np.ndarray, Wsub_mask:np.ndarray, Wsub_name_idx_map:dict, 
           num_top_words:int, rows_name:str,
           save_path:str, figsize2):
    plt.figure(dpi=300, figsize=figsize2)

    sns.heatmap(W_sub, 
                cmap="YlOrRd",
                #square=True, norm=LogNorm(),
                vmin=0,
                annot=True,
                linewidths=1,
                linecolor='black',
                cbar_kws={'label': 'Probability'},
                annot_kws={
                            "fontsize":10,
                            #"fontweight":"bold"
                            },
                #xticklabels=4,
                #fmt=".2f",
                mask=Wsub_mask # Uncomment if want full heatmap
                )

    plt.title(f'W Top {num_top_words} Features Per Cluster')
    plt.xlabel("Clusters")
    plt.ylabel(rows_name)

    plt.yticks(np.arange(0, len(Wsub_name_idx_map.keys())) + 0.5, 
                Wsub_name_idx_map.keys(), 
                rotation=0,
                #ha="center"
                )

    plt.tight_layout()
    plt.savefig(f'{save_path}/W_probability_heatmap_top{num_top_words}.png')
    plt.close()


def get_W_sub(words:np.ndarray, probabilities:np.ndarray, save_path:str, rows_name:str, num_top_words:int):

    all_unique_gene_names = set()
    for col_idx in range(words.shape[1]):
        row = words[:,col_idx]
        all_unique_gene_names |= set(row)

    Wsub_name_idx_map = {}
    for idx, name in enumerate(all_unique_gene_names):
        Wsub_name_idx_map[name] = idx
    W_sub = np.zeros((len(Wsub_name_idx_map), probabilities.shape[1]))
    
    for row_idx in range(words.shape[0]):
        for col_idx in range(words.shape[1]):
            gene = words[row_idx, col_idx]
            expression = probabilities[row_idx, col_idx]
            gene_idx = Wsub_name_idx_map[gene]
            W_sub[gene_idx, col_idx] = expression
            
    Wsub_mask = np.zeros(W_sub.shape).astype("bool")
    for row, col in np.argwhere(W_sub == 0):
        Wsub_mask[row,col] = True


    col_names = [rows_name]
    for k in range(W_sub.shape[1]):
        col_names.append(f'Cluster {k}')
    W_sub_df = pd.DataFrame(np.hstack([np.array(list(Wsub_name_idx_map.keys())).reshape(-1,1), W_sub]), columns=col_names)

    W_sub_df.to_csv(f'{save_path}/W_probability_table_top{num_top_words}.csv', index=False)

    return Wsub_name_idx_map, W_sub, Wsub_mask