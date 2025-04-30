import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from ...helpers.embeddings import (compute_doc_embedding, produce_label,
                                   compute_embeddings, closest_embedding_to_centroid,
                                   compute_centroids)

MODEL = 'SCINCL'  # either 'SCINCL' or 'SPECTER'
DISTANCE_METRIC = 'cosine'  # either 'cosine' or 'euclidean'
CENTER_METRIC = 'centroid'  # either 'centroid' or 'medoid'
NUM_TRIALS = 3
TEXT_COLS = ['title', 'abstract']
    
def label_clusters(top_words_df, api_key, n_trials=1):
    output = {}
    for col in tqdm(top_words_df.columns):
        col_key = int(col)
        if col_key not in output:
            output[col_key] = []
        words = top_words_df[col].to_list()

        for _ in range(n_trials):
            label = produce_label(api_key, words)
            output[col_key].append(label)
    return output

def process_annotations(annotations, max_tokens=5):
    filtered_annotations = [a for a in annotations if len(a.split()) <= max_tokens]
    filtered_annotations = [' '.join(token.capitalize() for token in a.split()) for a in filtered_annotations]
    return filtered_annotations

def label_clusters_openAI(top_words_df, api_key, open_ai_model, embedding_model, df):
    embeddings = compute_embeddings(df, model_name=embedding_model, cols=TEXT_COLS, use_gpu=False)
    centroids = compute_centroids(embeddings, df)
    centers = {k: closest_embedding_to_centroid(embeddings, c, metric=DISTANCE_METRIC) for k, c in centroids.items()}
    cluster_labels = label_clusters(top_words_df, api_key, n_trials=NUM_TRIALS)
    filtered_annotations = {k: process_annotations(v) for k, v in cluster_labels.items()}

    annotations = {}
    for cluster, cluster_annotations in tqdm(filtered_annotations.items(), total=len(filtered_annotations)):

        if cluster not in centers:
            print(f"Skipping cluster {cluster} — not found in computed centers.")
            continue

        cluster_center = centers[cluster][1]

        cluster_annotation_embeddings = []
        for ca in cluster_annotations:
            cluster_annotation_embeddings.append(compute_doc_embedding(ca, model_name=MODEL))

        distances = pairwise_distances(
            np.array(cluster_annotation_embeddings), [cluster_center], metric=DISTANCE_METRIC
        ).flatten()

        print(f"Cluster {cluster} distances and annotations:", list(zip(distances, cluster_annotations)))

        best_annotation = cluster_annotations[np.argmin(distances)]
        annotations[int(cluster)] = best_annotation

    return annotations
