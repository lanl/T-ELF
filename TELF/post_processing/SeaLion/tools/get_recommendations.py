import numpy as np
import pandas as pd
import os
import warnings
import networkx as nx
from pyvis.network import Network
import shutil


def _get_dfs(Xtilda, H, cols, rows, num_top_recommendations, recommend_probabilities):
    all_df_names = []
    all_df_scores = []
    H_clustering = np.argmax(H, axis=0)
    for k in range(H.shape[0]):
        indices = np.argwhere(H_clustering==k).flatten()
        samples = cols[indices]
        recommendation_indices = np.argsort(Xtilda[:,indices], axis=0)[::-1][:num_top_recommendations, :]
        recommendation_scores = np.zeros(recommendation_indices.shape).astype("object")
        recommendation_names = np.zeros(recommendation_indices.shape).astype("object")

        for xx, yy in enumerate(recommendation_indices):
            for x, y in enumerate(yy):
                score = Xtilda[:,indices][y,x]
                name = rows[y]
                if score == 0:
                    score = 0
                    name = "-"
                recommendation_scores[xx, x] = score
                recommendation_names[xx, x] = name
        
        curr_df_names = pd.DataFrame(np.vstack([samples, recommendation_names]))
        curr_df_scores = pd.DataFrame(np.vstack([samples, recommendation_scores]))

        headers_names = curr_df_names.iloc[0]
        new_curr_df_names  = pd.DataFrame(curr_df_names.values[1:], columns=headers_names)

        headers_scores = curr_df_scores.iloc[0]
        new_curr_df_scores  = pd.DataFrame(curr_df_scores.values[1:], columns=headers_scores)

        if recommend_probabilities:
            eps = np.finfo(float).eps
            new_curr_df_scores = np.divide(new_curr_df_scores, new_curr_df_scores.sum(axis=0) + eps)

        all_df_names.append(new_curr_df_names)
        all_df_scores.append(new_curr_df_scores)

    return all_df_names, all_df_scores

def recommendations_masked_all(UNKNOWN_MASK:np.ndarray, KNOWN_MASK:np.ndarray, W:np.ndarray, H:np.ndarray, S:np.ndarray,
                           bi:np.ndarray, bu:np.ndarray, global_mean:int, 
                           rows:np.ndarray, cols:np.ndarray,  num_top_recommendations:int,
                           recommend_probabilities:bool):
    
    if S is not None:
        Xtilda = np.add(np.add(W@S@H, bi).T, bu).T + global_mean
    else:
        Xtilda = np.add(np.add(W@H, bi).T, bu).T + global_mean

    Xtilda_unknown_masked = np.zeros(Xtilda.shape).astype("float32")
    Xtilda_known_masked = np.zeros(Xtilda.shape).astype("float32")
    for x, y in UNKNOWN_MASK:
        Xtilda_unknown_masked[x,y] = Xtilda[x,y]
    for x, y in KNOWN_MASK:
        Xtilda_known_masked[x,y] = Xtilda[x,y]

    if num_top_recommendations > Xtilda_unknown_masked.shape[0]:
        num_top_recommendations = Xtilda_unknown_masked.shape[0]
    
    all_df_names_unknown, all_df_scores_unknown = _get_dfs(Xtilda_unknown_masked, H, cols, rows, num_top_recommendations, recommend_probabilities)
    all_df_names_known, all_df_scores_known = _get_dfs(Xtilda_known_masked, H, cols, rows, num_top_recommendations, recommend_probabilities)

    return all_df_names_unknown, all_df_scores_unknown, all_df_names_known, all_df_scores_known

def recommendation_graph(UNKNOWN_MASK:np.ndarray, KNOWN_MASK:np.ndarray, W:np.ndarray, H:np.ndarray, S:np.ndarray,
                        bi:np.ndarray, bu:np.ndarray, global_mean:int, 
                        rows:np.ndarray, cols:np.ndarray,  num_top_recommendations:int,
                        recommend_probabilities:bool, save_path:str, edge_weight_multiplier:int):
    
    all_df_names_unknown, all_df_scores_unknown, all_df_names_known, all_df_scores_known = recommendations_masked_all(
            UNKNOWN_MASK=UNKNOWN_MASK,
            KNOWN_MASK=KNOWN_MASK,
            W=W,
            H=H,
            S=S,
            bi=bi,
            bu=bu,
            global_mean=global_mean,
            rows=rows,
            cols=cols,
            num_top_recommendations=num_top_recommendations,
            recommend_probabilities=recommend_probabilities
    )
    for K in range(H.shape[0]):
        G = nx.Graph()
        names_unknown = all_df_names_unknown[K]
        scores_unknown = all_df_scores_unknown[K]
        columns = scores_unknown.columns

        for col in columns:
            G.add_node(col, size=10, label=str(col), group=1)

        for col in columns:
            curr_names = list(names_unknown[col])
            curr_scores = list(scores_unknown[col])
            
            for nn, ss in zip(curr_names, curr_scores):
                if nn == "-":
                    continue
                G.add_node(nn, size=10, label=str(nn), group=2)
                G.add_edge(col, nn, weight=str(ss*edge_weight_multiplier), color="red", group=2)
        names_known = all_df_names_known[K]
        scores_known = all_df_scores_known[K]

        for col in columns:
            curr_names = list(names_known[col])
            curr_scores = list(scores_known[col])
            
            for nn, ss in zip(curr_names, curr_scores):
                if nn == "-":
                    continue
                G.add_node(nn, size=10, label=nn, group=2)
                G.add_edge(col, nn, weight=str(ss*edge_weight_multiplier), group=1)

        nt = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", 
             select_menu=True,
             filter_menu=True
             )
        nt.from_nx(G)
        nt.show_buttons(filter_=['physics', 'nodes', 'edges'])
        nt.save_graph(f'{os.path.join(save_path, str(K))}/recommendation_graph_{K}.html')
        shutil.copytree("lib", f'{os.path.join(save_path, str(K))}/lib')
        shutil.rmtree("lib")

def recommendations_masked(UNKNOWN_MASK:np.ndarray, KNOWN_MASK:np.ndarray, 
                           W:np.ndarray, H:np.ndarray, S:np.ndarray,
                           bi:np.ndarray, bu:np.ndarray, global_mean:int, 
                           rows:np.ndarray, cols:np.ndarray, save_path:str, num_top_recommendations:int,
                           recommend_probabilities:bool,
                           cols_name:str, rows_name:str):
    
    if S is not None:
        Xtilda = np.add(np.add(W@S@H, bi).T, bu).T + global_mean
    else:
        Xtilda = np.add(np.add(W@H, bi).T, bu).T + global_mean

    if recommend_probabilities:
        add_name = "probability_scores"
    else:
        add_name = "reconstruction_scores"

    Xtilda_unknown_masked = np.zeros(Xtilda.shape).astype("float32")
    Xtilda_known_masked = np.zeros(Xtilda.shape).astype("float32")
    for x, y in UNKNOWN_MASK:
        Xtilda_unknown_masked[x,y] = Xtilda[x,y]
    for x, y in KNOWN_MASK:
        Xtilda_known_masked[x,y] = Xtilda[x,y]

    # get top recommendations first
    all_top_xidx, all_top_yindx = __largest_indices(Xtilda_unknown_masked, num_top_recommendations)
    all_top_preds = []
    for x,y in zip(all_top_xidx, all_top_yindx):
        a, b, value = rows[x], cols[y], Xtilda_unknown_masked[x,y]
        if value > 0:
            all_top_preds.append((a, b, value))
    pd.DataFrame(all_top_preds, columns=[rows_name, cols_name, "score"]).to_csv(f'{save_path}/overall_top_{num_top_recommendations}_predictions.csv', index=False)

    # do for each topic
    if num_top_recommendations > Xtilda_unknown_masked.shape[0]:
        num_top_recommendations = Xtilda_unknown_masked.shape[0]
        warnings.warn("More recommendations requested than exist in the matrix. Reverting to maximum.")
    
    all_df_names_unknown, all_df_scores_unknown = _get_dfs(Xtilda_unknown_masked, H, cols, rows, num_top_recommendations, recommend_probabilities)
    all_df_names_known, all_df_scores_known = _get_dfs(Xtilda_known_masked, H, cols, rows, num_top_recommendations, recommend_probabilities)
    
    for k, (new_curr_df_names, new_curr_df_scores) in enumerate(zip(all_df_names_unknown, all_df_scores_unknown)):
        new_curr_df_names.to_csv(f'{os.path.join(save_path, str(k))}/UNKNOWN_top_{num_top_recommendations}_recommendations_names_{k}.csv', index=True)
        new_curr_df_scores.to_csv(f'{os.path.join(save_path, str(k))}/UNKNOWN_top_{num_top_recommendations}_recommendations_{add_name}_{k}.csv', index=True)

    for k, (new_curr_df_names, new_curr_df_scores) in enumerate(zip(all_df_names_known, all_df_scores_known)):
        new_curr_df_names.to_csv(f'{os.path.join(save_path, str(k))}/KNOWN_top_{num_top_recommendations}_recommendations_names_{k}.csv', index=True)
        new_curr_df_scores.to_csv(f'{os.path.join(save_path, str(k))}/KNOWN_top_{num_top_recommendations}_recommendations_{add_name}_{k}.csv', index=True)
        

def recommendations(W:np.ndarray, H:np.ndarray, S:np.ndarray,
                    bi:np.ndarray, bu:np.ndarray, global_mean:int, 
                    rows:np.ndarray, cols:np.ndarray, save_path:str, num_top_recommendations:int,
                    recommend_probabilities:bool):

    if S is not None:
        Xtilda = np.add(np.add(W@S@H, bi).T, bu).T + global_mean
    else:
        Xtilda = np.add(np.add(W@H, bi).T, bu).T + global_mean

    if recommend_probabilities:
        add_name = "probability_scores"
    else:
        add_name = "reconstruction_scores"
    
    if num_top_recommendations > Xtilda.shape[0]:
        num_top_recommendations = Xtilda.shape[0]
        warnings.warn("More recommendations requested than exist in the matrix. Reverting to maximum.")

    all_df_names, all_df_scores = _get_dfs(Xtilda, H, cols, rows, num_top_recommendations, recommend_probabilities)
    
    for k, (new_curr_df_names, new_curr_df_scores) in enumerate(zip(all_df_names, all_df_scores)):
        new_curr_df_names.to_csv(f'{os.path.join(save_path, str(k))}/ALL_top_{num_top_recommendations}_recommendations_names_{k}.csv', index=True)
        new_curr_df_scores.to_csv(f'{os.path.join(save_path, str(k))}/ALL_top_{num_top_recommendations}_recommendations_{add_name}_{k}.csv', index=True)


def __largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)