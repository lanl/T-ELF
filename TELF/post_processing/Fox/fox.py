# fox/core.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from .clustering_analyzer import ClusteringAnalyzer
from .visualizer import VisualizationManager
from ...helpers.file_system import check_path
from .openAI_summaries import label_clusters_openAI

class Fox:
    def __init__(self, summary_model=None, api_key=None, verbose=False, debug=False):
        self.summary_model = summary_model
        self.api_key = api_key
        self.verbose = verbose
        self.debug = debug

        self.cluster_analyzer = ClusteringAnalyzer(verbose=verbose)
        self.summarizer = ClusterSummarizer(api_key=api_key, summary_model=summary_model)
        self.visualizer = VisualizationManager(verbose=verbose)
        self.dir_manager = DirectoryManager()

    def post_process(self, npz_path: str, vocabulary_path: str, src_decomp_data_path: str,
                     output_dir: str = None, top_words_per_cluster: int = 50,
                     clean_cols_name: str = "clean_title_abstract", terms: list = None) -> str:

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f'File "{npz_path}" not found.')
        if not os.path.exists(vocabulary_path):
            raise FileNotFoundError('Vocabulary file not found!')
        if not os.path.exists(src_decomp_data_path):
            raise FileNotFoundError('Source data file for decomposition not found!')

        df = pd.read_csv(src_decomp_data_path)
        if clean_cols_name not in df:
            raise ValueError(f'{clean_cols_name} column is missing in the source data.')

        k = npz_path.split(".npz")[0].split("=")[-1]
        output_dir = output_dir or os.path.join(os.path.dirname(npz_path), f"k={k}")
        output_dir = check_path(output_dir)
        archive_dir = os.path.join(output_dir, 'archive')
        os.makedirs(archive_dir, exist_ok=True)

        data = np.load(npz_path, allow_pickle=True)
        W, H = data['W'], data['H']

        vocab = np.array(open(vocabulary_path).read().splitlines())

        post_processed_df_path = self.cluster_analyzer.analyze_clusters(
            H, W, vocab, df, output_dir, archive_dir,
            top_words_per_cluster, clean_cols_name, terms)

        return post_processed_df_path

    def post_process_stats(self, processing_path: str, clean_cols_name: str = "clean_title_abstract"):
        self.visualizer.generate_all_statistics(processing_path, clean_cols_name)

    def makeSummariesAndLabelsOpenAi(self, processing_path: str):
        self.summarizer.generate_labels_and_summaries(processing_path)

    def rename_cluster_dirs_from_stats(self, processing_path: str):
        self.dir_manager.rename_from_stats(processing_path)

    def getApiKey(self):
        return self.api_key

    def setApiKey(self, api_key):
        self.api_key = api_key

    def getSummaryModel(self):
        return self.summary_model

    def setSummaryModel(self, summary_model):
        self.summary_model = summary_model




class ClusterSummarizer:
    def __init__(self, api_key=None, summary_model=None):
        self.api_key = api_key
        self.summary_model = summary_model

    def generate_labels_and_summaries(self, processing_path: str):
        df = pd.read_csv(processing_path)
        base_dir = Path(processing_path).parent

        cluster_summaries = {c: '' for c in sorted(df.cluster.dropna().unique())}

        # Get labels via OpenAI
        top_words_df = pd.read_csv(base_dir / 'top_words.csv')
        cluster_labels = label_clusters_openAI(
            top_words_df,
            api_key=self.api_key,
            open_ai_model=self.summary_model,
            embedding_model='SCINCL',
            df=df
        )

        # Combine into dataframe
        summary_df = pd.DataFrame({
            'cluster': list(cluster_summaries.keys()),
            'label': [cluster_labels.get(k, '') for k in cluster_summaries.keys()],
            'summary': [cluster_summaries[k] for k in cluster_summaries.keys()]
        })

        summary_df.to_csv(base_dir / 'cluster_summaries.csv', index=False)


class DirectoryManager:
    def __init__(self):
        pass

    def rename_from_stats(self, post_processed_df_path: str):
        """
        Renames cluster directories using label and document count from stats.csv.

        Format:
            <cluster>-<label>_<num_papers>-documents
            OR if no label:
            <cluster>-unlabeled_<num_papers>-documents
        """
        base_dir = Path(post_processed_df_path).parent
        stats_path = base_dir / 'stats.csv'

        if not stats_path.exists():
            raise FileNotFoundError(f"Could not find stats.csv at {stats_path}")

        stats_df = pd.read_csv(stats_path)

        # Ensure clusters are numeric and not NaN
        stats_df = stats_df[pd.to_numeric(stats_df['cluster'], errors='coerce').notna()]
        stats_df['cluster'] = stats_df['cluster'].astype(float)

        for _, row in stats_df.iterrows():
            cluster_float = row['cluster']
            cluster_name = str(int(cluster_float))  # "3.0" → "3"

            cluster_dir = base_dir / cluster_name
            if not cluster_dir.exists():
                print(f"[WARN] Cluster directory not found: {cluster_dir}")
                continue

            # Label logic
            label = str(row.get('label', '')).strip()
            if not label or label.lower() == 'nan':
                label = 'unlabeled'
            else:
                label = label[:30].replace(' ', '_')

            # Document count logic
            num_papers = row.get('num_papers', 0)
            if pd.isna(num_papers):
                num_papers = 0

            new_name = f"{cluster_name}-{label}_{int(num_papers)}-documents"
            new_dir = base_dir / new_name

            try:
                cluster_dir.rename(new_dir)
                print(f"[OK] Renamed {cluster_dir.name} → {new_name}")
            except Exception as e:
                print(f"[ERROR] Failed to rename {cluster_dir} → {new_dir}: {e}")
