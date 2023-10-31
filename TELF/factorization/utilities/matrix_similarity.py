import os, numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel,delayed
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy import stats

SIMILARITY_WORD_THRESHOLD = .75
STATS_INDEX = 0


def longest_common_subsequence(word1, word2):
    m, n = len(word1), len(word2)
    lcs_table = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                lcs_table[i][j] = 1 + lcs_table[i - 1][j - 1]
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    
    lcs_length = lcs_table[m][n]
    similarity = lcs_length / max(m, n)
    if similarity < SIMILARITY_WORD_THRESHOLD:
        similarity = 0 
    return similarity
 

def compute_cost_parallel(i_col, j_col):
    z_temp_score = 0
    for word_of_i in i_col:
        for word_of_j in j_col:
             z_temp_score += (1 - longest_common_subsequence(word_of_i, word_of_j))
    return z_temp_score / (len(i_col) * len(j_col))


def compare_decompositions(path1, path2, rejection_threshold = .4, save_scores=True):
    if not os.path.exists(path1):
        raise FileNotFoundError(f"The specified path '{path1}' does not exist.")
    if not os.path.exists(path2):
        raise FileNotFoundError(f"The specified path '{path2}' does not exist.")

    
    if 'csv' in path1 and 'csv' in path2: # Dissimilar vocabularies, read words
        fewer_cols = pd.read_csv(path1)
        more_cols = pd.read_csv(path2)
        
        if len(more_cols.columns) < len(fewer_cols.columns): # csv with more columns needs to be path2  
            fewer_cols, more_cols = more_cols, fewer_cols
        
        slices = Parallel(n_jobs=-1)\
            (delayed(compute_cost_parallel)(fewer_cols[str(i)], more_cols[str(j)]) \
            for i in range(len(fewer_cols.columns)) for j in range(len(more_cols.columns))
        )
        cost_matrix_unshaped = np.array(slices)
        cost_matrix = cost_matrix_unshaped.reshape(len(fewer_cols.columns), len(more_cols.columns))
    
    elif 'npz' in path1 and 'npz' in path2: # Same vocabularies, read W
        npz1, npz2 = np.load(path1), np.load(path2)
        npz1_w, npz2_w = npz1["W"],  npz2["W"]
        
        if npz1_w.shape[1] < npz2_w.shape[1]: # Larger npz should be in npz1
            npz1_w, npz2_w = npz2_w, npz1_w
            
        cost_matrix = np.zeros((len(npz2_w[0]), len( npz1_w [0])))
        for i in range(len(npz2_w [0])):
            for j in range(len(npz1_w[0])):
                column_1 = npz2_w [:, i]  
                column_2 = npz1_w[:, j]  
                dist = 1 -  stats.pearsonr(column_1, column_2)[STATS_INDEX]
                cost_matrix[i][j] = dist
     
    else:
        print("Nonstandard data format, exiting comparison. " +\
              "Need csvs of top words or npz files containing W of NMFK decomposition.")
        return None

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    data = [(row, col) for row, col in zip(list(row_ind), list(col_ind))]

    df = pd.DataFrame(data, columns=["columns of file1", "columns of file2"])
    global_min, global_max = cost_matrix.min(), cost_matrix.max()
    scaled_data = (cost_matrix - global_min) / (global_max - global_min)

    df_scores = pd.DataFrame(scaled_data)
    df_reordered = df_scores[df_scores.columns[list(col_ind)]]
    df_reordered.to_csv('similarities_scoring_source.csv',index=False)

    diagonal_values = np.diagonal(df_reordered.values)

    rejection_indicies = []
    for i in range(len(diagonal_values)):
        if diagonal_values[i] > rejection_threshold:
            rejection_indicies.append(i)

    df.loc[rejection_indicies, df.columns[1]] = 'No similarity'
    csv_file = 'similarities_scoring.csv'
    df.to_csv(csv_file, index=False)


    plt.figure(figsize=(8,6))
    sns.heatmap(df_reordered, annot=True, cmap='viridis_r', cbar_kws={'label': 'Value'})
