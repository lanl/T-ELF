import string
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import pairwise_distances
from langchain_ollama import OllamaLLM
from openai import OpenAI


class ClusterLabeler:

    MODELS = {
        'SCINCL': 'malteos/scincl',
        'SPECTER': 'allenai/specter2_base'
    }

    def __init__(
        self,
        embedding_model: str = "SCINCL",
        distance_metric: str = "cosine",
        center_metric: str = "centroid",
        text_cols: list = None,
        num_trials: int = 10
    ):
        self.embedding_model = embedding_model
        self.distance_metric = distance_metric
        self.center_metric = center_metric
        self.num_trials = num_trials
        self.text_cols = text_cols if text_cols else ["title", "abstract"]

    # ------------------------------------------------
    # Embedding / Centroid
    # ------------------------------------------------
    def initialize_model_and_device(self, embedding_model: str, embedds_use_gpu: bool = False):
        device = 'cuda:0' if embedds_use_gpu and torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(self.MODELS[embedding_model])
        model = AutoModel.from_pretrained(self.MODELS[embedding_model]).to(device)
        return tokenizer, model, device

    def compute_embeddings_helper(self, text, *, model, tokenizer, device, max_length=512):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(device)
        out = model(**inputs)
        emb = torch.mean(out.last_hidden_state[0], dim=0)
        return emb.detach().cpu().numpy()

    def compute_embeddings(self, df, *, model, tokenizer, device, cols=None, sep_token='[SEP]'):
        if cols is None:
            cols = self.text_cols
        papers = df[cols].fillna('').apply(lambda row: sep_token.join(row), axis=1).reset_index()
        papers.columns = ['id', 'text']

        embeddings = {}
        for _, row in tqdm(papers.iterrows(), total=len(papers)):
            embeddings[row['id']] = self.compute_embeddings_helper(
                row['text'], model=model, tokenizer=tokenizer, device=device
            )
        return embeddings

    def compute_centroids(self, embeddings_dict, df):
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

    def closest_embedding_to_centroid(self, embeddings_dict, centroid_vec, metric='cosine'):
        """
        Return (min_index, centroid_vec), exactly as your snippet.
        """
        arr = np.array(list(embeddings_dict.values()))
        dist = pairwise_distances(arr, [centroid_vec], metric=metric)
        min_index = np.argmin(dist)
        return (min_index, centroid_vec)

    # ------------------------------------------------
    # Prompting, LLM, Labeling
    # ------------------------------------------------
    def criteria_and_info_to_prompt(self, base_prompt, additional_information=None, criteria=None):
        prompt = base_prompt
        if additional_information:
            prompt += "\nHere is additional information:\n"
            for k, v in additional_information.items():
                prompt += f"{k}: {v}\n"

        if criteria:
            prompt += "\nHere are additional criteria. Make sure your response is a single phrase that follows the criteria:\n"
            for k, v in criteria.items():
                prompt += f"{k}: {v}\n"
        prompt += (
            "\nMake sure your response is only the label without any commentary or additional formatting "
            "(for instance, do not wrap the answer in quotes). Do not mention the criteria in the label."
        )
        return prompt

    def validate_llm_output(self, output, criteria):
        if not criteria:
            return True
        words = output.split()
        wc = len(words)
        if "minimum words" in criteria and wc < criteria["minimum words"]:
            return False
        if "maximum words" in criteria and wc > criteria["maximum words"]:
            return False
        if "must_contain" in criteria:
            for w in criteria["must_contain"]:
                if w not in output:
                    return False
        if "must_not_contain" in criteria:
            for w in criteria["must_not_contain"]:
                if w in output:
                    return False
        return True

    def generate_valid_labels(self, llm, prompt, criteria, number_of_labels):
        labels = []
        while len(labels) < number_of_labels:
            candidate = llm.invoke(prompt).strip()
            if self.validate_llm_output(candidate, criteria):
                labels.append(candidate)
        return labels

    def generate_labels_with_criteria(
        self,
        llm,
        input_words,
        additional_information=None,
        criteria=None,
        number_of_labels=10
    ):
        prompt = f"Create a concise label for the following words. Order matters, the more important words are first: {input_words}."
        prompt = self.criteria_and_info_to_prompt(prompt, additional_information, criteria)
        return self.generate_valid_labels(llm, prompt, criteria, number_of_labels)

    def produce_label(self, client: OpenAI, words: str):
        prompt = (
            f"Use the following words to establish a theme or a topic: {', '.join(words)!r}. "
            "The output should be a label that has at most 3 tokens and is characterized by the provided words. "
            "The output should not be in quotes."
        )
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=2048
        )
        label = response.choices[0].text.strip()
        return label.strip().translate(str.maketrans('', '', string.punctuation + '\n'))

    def labels_from_models(
        self,
        input_words,
        additional_information=None,
        criteria=None,
        number_of_labels=10,
        models=None,
        api_keys=None
    ):
        if models is None:
            models = {'ollama': ['llama3.2:3b-instruct-fp16']}
        if api_keys is None:
            api_keys = {'openai': None}

        all_labels = []

        # 1) Ollama
        if models.get('ollama'):
            for model_name in models['ollama']:
                llm = OllamaLLM(model=model_name, base_url="http://localhost:11434")
                new_labels = self.generate_labels_with_criteria(
                    llm=llm,
                    input_words=input_words,
                    additional_information=additional_information,
                    criteria=criteria,
                    number_of_labels=number_of_labels
                )
                all_labels.extend(new_labels)

        # 2) openai
        if models.get('openai'):
            openai_api_key = api_keys.get('openai')
            client = OpenAI(api_key=openai_api_key)
            for _ in range(number_of_labels):
                candidate = self.produce_label(client, input_words)
                if self.validate_llm_output(candidate, criteria):
                    all_labels.append(candidate)

        return all_labels


    def label_clusters_ollama(
        self,
        top_words_df: pd.DataFrame,
        ollama_model_name: str,
        embedding_model: str,
        df: pd.DataFrame,
        criteria: dict = None,
        additional_information: dict = None,
        number_of_labels: int = 10,
        embedds_use_gpu: bool = False
    ):
        tokenizer, model_obj, device = self.initialize_model_and_device(embedding_model, embedds_use_gpu)
        doc_embeddings = self.compute_embeddings(df, model=model_obj, tokenizer=tokenizer, device=device)
        centroids = self.compute_centroids(doc_embeddings, df)

        # e.g. centers[0] = (min_idx, centroid_vec)
        centers = {}
        for cid, centroid_vec in centroids.items():
            if centroid_vec is not None:
                centers[cid] = self.closest_embedding_to_centroid(doc_embeddings, centroid_vec, metric=self.distance_metric)

        # Step 2: For each column -> generate candidate labels
        cluster_labels = {}
        for col in tqdm(top_words_df.columns):
            words = top_words_df[col].to_list()
            cluster_labels[col] = self.labels_from_models(
                input_words=words,
                additional_information=additional_information,
                criteria=criteria,
                number_of_labels=number_of_labels,
                models={'ollama': [ollama_model_name]}
            )

        # Step 3: pick best label
        annotations = {}
        for col_str, candidate_labels in tqdm(cluster_labels.items(), total=len(cluster_labels)):
            cluster_id = int(col_str)

            if cluster_id not in centers:
                print(f"Skipping cluster_id={cluster_id} because it's not in centers.")
                continue

            _, centroid_vec = centers[cluster_id]  # (closest_idx, centroid)
            label_embs = []
            for lbl in candidate_labels:
                emb = self.compute_embeddings_helper(lbl, model=model_obj, tokenizer=tokenizer, device=device)
                label_embs.append(emb)
            label_embs = np.array(label_embs)

            # Distances
            dists = pairwise_distances(label_embs, [centroid_vec], metric=self.distance_metric).flatten()
            best_idx = np.argmin(dists)
            best_label = candidate_labels[best_idx]
            annotations[cluster_id] = best_label

        return annotations
