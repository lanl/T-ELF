from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def H_clustering(H:np.ndarray, S:np.ndarray, save_path:str, cols_name:str, cols:list, figsize1):

    H_clustering = __get_H_clustering(H)
    __H_clustering_table(H_clustering, save_path, cols_name, cols)
    __H_clustering_plot(H, H_clustering, save_path, cols_name, cols, figsize1)
    __H_clustering_custom(H, H_clustering, save_path)
    __samples_in_clusters(H, H_clustering, save_path, cols)
    __centroids(H, S, H_clustering, save_path)
    __sample_cluster_weight_plot(H, S, H_clustering, save_path, cols, cols_name, figsize1)
    __H_hierarchical_clustering_dendogram(H, H_clustering, save_path, cols, cols_name, figsize1)

def __H_hierarchical_clustering_dendogram(H:np.ndarray, H_clustering, save_path:str, cols:str, cols_name:str, figsize1):
    cols_cluster = []
    for idx, cc in enumerate(cols):
        cols_cluster.append(f'{cc} ({H_clustering[idx]})')
    cols_cluster = np.array(cols_cluster)

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage="complete")
    model = model.fit(H.T)
    __plot_dendrogram(model=model, cols_name=cols_name, cols=cols_cluster, save_path=save_path, figsize1=figsize1)

def __sample_cluster_weight_plot(H:np.ndarray, S:np.ndarray, H_clustering:np.ndarray, save_path:str, cols:str, cols_name:str, figsize1):
    H = H.copy()
    cols_cluster = []
    for idx, cc in enumerate(cols):
        cols_cluster.append(f'{cc} ({H_clustering[idx]})')
    cols_cluster = np.array(cols_cluster)

    # if we have a mixing matrix
    if S is not None:
        SH = (S@H)
    else:
        SH = H

    sort_by_cluster = np.argsort(H_clustering, axis=0)
    SH = np.divide(SH, np.sum(SH, axis=0)) # normalize so each column sums up to 1
    Hdf = pd.DataFrame(SH[:,sort_by_cluster]).T
    plt.set_cmap('tab20')
    
    Hdf.plot(kind='bar', stacked=True, figsize=figsize1, colormap="tab20")
    plt.title(f'{cols_name} Cluster Weights')
    plt.xlabel(f'{cols_name}')
    plt.ylabel("Weight")
    plt.xticks(np.arange(0, len(cols_cluster[sort_by_cluster]), 1), cols_cluster[sort_by_cluster], rotation=90)
    plt.tight_layout()
    plt.savefig(f'{save_path}/{cols_name}_cluster_weights.png')
    plt.close()

    # now sort for each k
    for k in range(H.shape[0]):
        select_indices = np.argwhere(H_clustering == k).flatten()
        if len(select_indices) <= 0:
            continue
        sort_idx = np.argsort(H[:, select_indices][k], axis=0)[::-1]
        SH_select = SH[:,select_indices] # filder to samples in k
        Hsort_k = SH_select[:,sort_idx] # sort
        cols_cluster_select = cols_cluster[select_indices]

        Hdf = pd.DataFrame(Hsort_k).T
        Hdf.plot(kind='bar', stacked=True, figsize=figsize1, colormap="tab20")
        plt.title(f'{cols_name} Cluster Weights')
        plt.xlabel(f'{cols_name}')
        plt.ylabel("Weight")
        plt.xticks(np.arange(0, len(cols_cluster_select), 1), cols_cluster_select[sort_idx], rotation=90)
        plt.tight_layout()
        plt.savefig(f'{os.path.join(save_path, str(k))}/{cols_name}_{k}-sorted_cluster_weights.png')
        plt.close()


def __centroids(H:np.ndarray, S:np.ndarray, H_clustering:np.ndarray, save_path:str):

    # if we have a mixing matrix
    if S is not None:
        SH = (S@H)
    else:
        SH = H

    for k in range(H.shape[0]):
        indices = np.argwhere(H_clustering==k)
        centroids = np.mean(SH[:,indices], axis=1).flatten()
        plt.figure(dpi=300)
        plt.title(f'H Clustering Centroids - Cluster={k}')
        sns.barplot(centroids)
        plt.tight_layout()
        plt.savefig(f'{os.path.join(save_path, str(k))}/centroids_H_clustering_{k}.png')
        plt.close()

def __samples_in_clusters(H:np.ndarray, H_clustering:np.ndarray, save_path:str, cols:list):

    for k in range(H.shape[0]):
        indices = np.argwhere(H_clustering==k)
        pd.DataFrame(cols[indices].flatten(), columns=["samples"]).to_csv(f'{os.path.join(save_path, str(k))}/samples_in_cluster_{k}.csv', index=False)

def __H_clustering_custom(H:np.ndarray, H_clustering:np.ndarray, save_path:str):
    H_clustering_table = {"cluster":[], "counts":[], "percentage":[]}
    for k in range(H.shape[0]):
        H_clustering_table["cluster"].append(k)
        H_clustering_table["counts"].append(np.count_nonzero(H_clustering == k))
        H_clustering_table["percentage"].append(round(np.count_nonzero(H_clustering == k) / H.shape[1], 4))

    H_clustering_table_df = pd.DataFrame.from_dict(H_clustering_table)
    H_clustering_table_df.to_csv(f'{save_path}/table_H-clustering.csv', index=False)

def __H_clustering_plot(H:np.ndarray, H_clustering:np.ndarray, save_path:str, cols_name:str, cols:list, figsize1):
    K = H.shape[0]
    H_clustering_array = np.zeros((K, len(H_clustering)))
    for k in range(K):
        H_clustering_array[k, np.argwhere(H_clustering == k).flatten()] = 1
        H_clustering_array[k, np.argwhere(H_clustering == k).flatten()] = 1

    plt.figure(dpi=300, figsize=figsize1)
    sns.heatmap(H_clustering_array, 
                cmap="YlOrRd",  
                vmin=0,
                annot=True
                )

    plt.title("H Clustering")
    plt.xlabel(cols_name)
    plt.ylabel("Clusters")

    plt.xticks(np.arange(0, H.shape[1]) + 0.5, cols, 
                ha="center", rotation=90)

    plt.tight_layout()
    plt.savefig(f'{save_path}H_clustering.png')
    plt.close()

def __H_clustering_table(H_clustering:np.ndarray, save_path:str, cols_name:str, cols:list):
    H_clustering_results = {}
    H_clustering_results[cols_name] = cols
    H_clustering_results["H_clustering"] = H_clustering
    H_clustering_df = pd.DataFrame(H_clustering_results)
    H_clustering_df.to_csv(f'{save_path}/H_clustering.csv', index=False)

def __get_H_clustering(H:np.ndarray):
    return np.argmax(H, axis=0)

def __plot_dendrogram(model, cols_name, cols, save_path, figsize1,**kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(dpi=300, figsize=figsize1)
    dendrogram(linkage_matrix, labels=cols, 
               distance_sort=True, show_leaf_counts=True, 
               leaf_rotation=90,
               **kwargs)
    plt.title("H Hierarchical Clustering Dendrogram")
    plt.xlabel(cols_name)
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(f'{save_path}/H_hierarchical_clustering_dendrogram.png')
    plt.close()
