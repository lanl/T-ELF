from .helpers.local_labels import ClusterLabeler  
from .helpers.intial_post_process import HNMFkPostProcessor
from .helpers.post_statistics import HNMFkStatsGenerator
from pathlib import Path
import pandas as pd

class ArcticFox:
    def __init__(
        self,
        model,
        embedding_model="SCINCL",
        distance_metric="cosine",
        center_metric="centroid",
        text_cols=None,
        top_n_words=50,
        clean_cols_name="clean_title_abstract",

        # Postprocessor column config
        col_year='year',
        col_type='type',
        col_cluster='cluster',
        col_cluster_coords='cluster_coordinates',
        col_similarity='similarity_to_cluster_centroid',
    ):
        self.model = model
        self.clean_cols_name = clean_cols_name

        self.labeler = ClusterLabeler(
            embedding_model=embedding_model,
            distance_metric=distance_metric,
            center_metric=center_metric,
            text_cols=text_cols
        )

        self.postprocessor = HNMFkPostProcessor(
            top_n_words=top_n_words,
            default_clean_col=clean_cols_name,
            col_year=col_year,
            col_type=col_type,
            col_cluster=col_cluster,
            col_cluster_coords=col_cluster_coords,
            col_similarity=col_similarity
        )

        self.stats_generator = HNMFkStatsGenerator(clean_cols_name=clean_cols_name)

        # Allow reuse of these columns in other methods
        self.col_cluster = col_cluster

    def run_full_pipeline(
        self,
        vocab,
        data_df,
        text_column=None,
        ollama_model="llama3.2:3b-instruct-fp16",
        label_clusters=True,
        generate_stats=True,
        generate_visuals=True,
        process_parents=True,
        skip_completed=True,
        label_criteria=None,
        label_info=None,
        number_of_labels=5
    ):
        text_column = text_column or self.clean_cols_name

        print("Step 1: Post-processing W/H matrix and cluster data...")
        self.postprocessor.post_process_hnmfk(
            hnmfk_model=self.model,
            V=vocab,
            D=data_df,
            col_name=text_column,
            skip_completed=skip_completed,
            process_parents=process_parents
        )

        if label_clusters:
            print("Step 2: Labeling clusters with LLM...")
            self._label_all_clusters(
                vocab=vocab,
                data_df=data_df,
                text_column=text_column,
                ollama_model=ollama_model,
                label_criteria=label_criteria,
                label_info=label_info,
                number_of_labels=number_of_labels,
                process_parents=process_parents
            )

        if generate_stats:
            print("Step 3: Generating Peacock visual stats...")
            self.stats_generator.generate_cluster_stats(
                model=self.model,
                process_parents=process_parents,
                skip_completed=skip_completed
            )


    def _label_all_clusters(
        self, vocab, data_df, text_column, ollama_model,
        label_criteria, label_info, number_of_labels, process_parents
    ):
        for node in self.model.traverse_nodes():
            if node["leaf"] or process_parents:
                w = node['W']
                if w is None:
                    w = node['signature'].reshape(-1, 1)

                node_dir = Path(node["node_save_path"]).resolve().parent
                cluster_file = node_dir / f"cluster_for_k={w.shape[1]}.csv"
                top_words_file = node_dir / "top_words.csv"

                if cluster_file.exists() and top_words_file.exists():
                    df = pd.read_csv(cluster_file)
                    top_words_df = pd.read_csv(top_words_file)

                    print(node_dir)

                    annotations = self.labeler.label_clusters_ollama(
                        top_words_df=top_words_df,
                        ollama_model_name=ollama_model,
                        embedding_model=self.labeler.embedding_model,
                        df=df,
                        criteria=label_criteria,
                        additional_information=label_info,
                        number_of_labels=number_of_labels,
                        embedds_use_gpu=False
                    )

                    pd.DataFrame([
                        {self.col_cluster: k, 'label': v, 'summary': ""}
                        for k, v in annotations.items()
                    ]).to_csv(node_dir / "cluster_summaries.csv", index=False)

    def run_labeling_only(self, df, top_words_df, ollama_model_name, label_criteria=None, additional_info=None, number_of_labels=5):
        return self.labeler.label_clusters_ollama(
            top_words_df=top_words_df,
            ollama_model_name=ollama_model_name,
            embedding_model=self.labeler.embedding_model,
            df=df,
            criteria=label_criteria,
            additional_information=additional_info,
            number_of_labels=number_of_labels
        )

    def run_postprocessing_only(self, V, D, col_name=None, **kwargs):
        return self.postprocessor.post_process_hnmfk(
            hnmfk_model=self.model,
            V=V,
            D=D,
            col_name=col_name or self.clean_cols_name,
            **kwargs
        )

    def run_stats_only(self, process_parents=True, skip_completed=True):
        return self.stats_generator.generate_cluster_stats(
            model=self.model,
            process_parents=process_parents,
            skip_completed=skip_completed
        )
