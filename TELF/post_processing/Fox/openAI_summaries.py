import string
import numpy as np
import warnings
import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import pairwise_distances

MODEL = 'SCINCL'  # either 'SCINCL' or 'SPECTER'
DISTANCE_METRIC = 'cosine'  # either 'cosine' or 'euclidean'
CENTER_METRIC = 'centroid'  # either 'centroid' or 'medoid'
NUM_TRIALS = 3
TEXT_COLS = ['title', 'abstract']

def compute_embeddings_helper(text, *, model=None, tokenizer=None, model_name='SCINCL', device='cpu', max_length=512):
    if model is None and tokenizer is None:
        mn = get_model_name(model_name)
        tokenizer = AutoTokenizer.from_pretrained(mn)
        model = AutoModel.from_pretrained(mn).to(device)
        
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(device)
    result = model(**inputs)
    
    if device == 'cpu':
        return torch.mean(result.last_hidden_state[0], dim=0).detach().numpy()
    else:
        return torch.mean(result.last_hidden_state[0], dim=0).cpu().detach().numpy()

def produce_label(api_key, words):
    client = OpenAI(api_key=api_key)
    prompt = f"Use the following words to establish a theme or a topic: {', '.join(words)!r}. The output should be a label that has at most 3 tokens and is characterized by the provided words. Ideally it should use one or more of the first few words in the list if possible. The output should not be in quotes"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=2048 
    )
    label = response.choices[0].text.strip()
    return label.strip().translate(str.maketrans('', '', string.punctuation + '\n'))

def get_model_name(model_name):
    assert model_name in ('SCINCL', 'SPECTER'), f'Unknown model name "{model_name}"!'
    if model_name == 'SCINCL':
        return 'malteos/scincl'
    if model_name == 'SPECTER':
        return 'allenai/specter2_base'
    
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

def compute_centroids(embeddings, df):
    centroids = {}
    for cluster, group in df.groupby('cluster'):
        cluster_embeddings = np.array([embeddings[idx] for idx in group.index if idx in embeddings])
        centroid = np.mean(cluster_embeddings, axis=0)
        centroids[cluster] = centroid
    return centroids

def compute_embeddings(df, *, model_name='SCINCL', cols=['title', 'abstract'], sep_token='[SEP]', as_np=False, use_gpu=True):
    papers = df[cols].fillna('').apply(lambda row: sep_token.join(row), axis=1).reset_index()
    papers.columns = ['id', 'text']
    device = 'cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu'
    if use_gpu and device == 'cpu':
        warnings.warn(f'Tried to use GPU, but GPU is not available. Using {device}.')

    mn = get_model_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(mn)
    model = AutoModel.from_pretrained(mn).to(device)
    embeddings = {}
    for _, row in tqdm(papers.iterrows(), total=len(papers)):
        embedding = compute_embeddings_helper(row['text'], model=model, tokenizer=tokenizer, device=device)
        embeddings[row['id']] = embedding

    if as_np:
        return np.array(list(embeddings.values()))
    return embeddings

def closest_embedding_to_centroid(embeddings, centroid, metric='cosine'):
    embeddings_array = np.array(list(embeddings.values()))
    distances = pairwise_distances(embeddings_array, [centroid], metric=metric)
    min_index = np.argmin(distances)
    return min_index, centroid

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
            cluster_annotation_embeddings.append(compute_embeddings_helper(ca, model_name=MODEL))

        distances = pairwise_distances(
            np.array(cluster_annotation_embeddings), [cluster_center], metric=DISTANCE_METRIC
        ).flatten()

        print(f"Cluster {cluster} distances and annotations:", list(zip(distances, cluster_annotations)))

        best_annotation = cluster_annotations[np.argmin(distances)]
        annotations[int(cluster)] = best_annotation

    return annotations
