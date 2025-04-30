import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from langchain_ollama import OllamaLLM

from ....helpers.embeddings import (compute_doc_embedding, produce_label,
                                    compute_embeddings, closest_embedding_to_centroid,
                                    compute_centroids)

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
            for _ in range(number_of_labels):
                candidate = produce_label(openai_api_key, input_words)
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
        doc_embeddings = compute_embeddings(df, model_name=self.embedding_model )
        centroids = compute_centroids(doc_embeddings, df)

        # e.g. centers[0] = (min_idx, centroid_vec)
        centers = {}
        for cid, centroid_vec in centroids.items():
            if centroid_vec is not None:
                centers[cid] = closest_embedding_to_centroid(doc_embeddings, centroid_vec, metric=self.distance_metric)

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
            device = 'cuda' if embedds_use_gpu else 'cpu'
            for lbl in candidate_labels:

                emb = compute_doc_embedding(lbl, model_name=self.embedding_model, device=device)
                label_embs.append(emb)
            label_embs = np.array(label_embs)

            # Distances
            dists = pairwise_distances(label_embs, [centroid_vec], metric=self.distance_metric).flatten()
            best_idx = np.argmin(dists)
            best_label = candidate_labels[best_idx]
            annotations[cluster_id] = best_label

        return annotations
