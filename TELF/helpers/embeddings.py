from transformers import AutoTokenizer, AutoModel
import torch
from openai import OpenAI
import string
import warnings
from tqdm import tqdm
import numpy as np
from sklearn.metrics import pairwise_distances

def get_model_name(model_name):
    assert model_name in ('SCINCL', 'SPECTER'), f'Unknown model name "{model_name}"!'
    if model_name == 'SCINCL':
        return 'malteos/scincl'
    if model_name == 'SPECTER':
        return 'allenai/specter2_base'

def initialize_model_and_device(embedding_model: str, use_gpu: bool = False):
    device = 'cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(get_model_name(embedding_model))
    model = AutoModel.from_pretrained(get_model_name(embedding_model)).to(device)
    return tokenizer, model, device


def compute_doc_embedding(text, *, model=None, tokenizer=None, model_name='SCINCL', device='cpu', max_length=512):
    use_gpu = True  
    if device == 'cpu': 
        use_gpu = False

    if model is None and tokenizer is None:
        tokenizer, model, device = initialize_model_and_device(model_name, use_gpu)
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(device)
    result = model(**inputs)
    
    if not use_gpu:
        return torch.mean(result.last_hidden_state[0], dim=0).detach().numpy()
    else:
        return torch.mean(result.last_hidden_state[0], dim=0).cpu().detach().numpy()
    

    
def produce_label(openai_api_key: str, words: str):
    prompt = (
        f"Use the following words to establish a theme or a topic: {', '.join(words)!r}. "
        "The output should be a label that has at most 3 tokens and is characterized by the provided words. "
        "The output should not be in quotes."
    )
    client = OpenAI(api_key=openai_api_key)
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=2048
    )
    label = response.choices[0].text.strip()
    return label.strip().translate(str.maketrans('', '', string.punctuation + '\n'))

def compute_embeddings(df, *, model_name='SCINCL', cols=['title', 'abstract'], sep_token='[SEP]', as_np=False, use_gpu=True):
    tokenizer, model, device = initialize_model_and_device(model_name, use_gpu)
    if use_gpu and device == 'cpu':
        warnings.warn(f'Tried to use GPU, but GPU is not available. Using {device}.')

    papers = df[cols].fillna('').apply(lambda row: sep_token.join(row), axis=1).reset_index()
    papers.columns = ['id', 'text']

    embeddings = {}
    for _, row in tqdm(papers.iterrows(), total=len(papers)):
        embedding = compute_doc_embedding(row['text'], model=model, tokenizer=tokenizer, device=device)
        embeddings[row['id']] = embedding

    if as_np:
        return np.array(list(embeddings.values()))
    return embeddings

def closest_embedding_to_centroid(embeddings, centroid, metric='cosine'):
    embeddings_array = np.array(list(embeddings.values()))
    distances = pairwise_distances(embeddings_array, [centroid], metric=metric)
    min_index = np.argmin(distances)
    return min_index, centroid

def compute_centroids(embeddings_dict, df):
    """
    Just like the snippet: compute mean embedding per cluster.
    """
    centroids = {}
    for cluster_id in df['cluster'].unique():
        idxs = df[df['cluster'] == cluster_id].index
        embs = [embeddings_dict[i] for i in idxs if i in embeddings_dict]
        if embs:
            centroids[cluster_id] = np.mean(embs, axis=0)
        else:
            centroids[cluster_id] = None
    return centroids
