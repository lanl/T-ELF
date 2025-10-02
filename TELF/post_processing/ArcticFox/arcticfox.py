from .helpers.local_labels import ClusterLabeler
from .helpers.intial_post_process import HNMFkPostProcessor
from .helpers.post_statistics import HNMFkStatsGenerator
from pathlib import Path
from typing import Optional, Sequence, Literal, Set
import pandas as pd

Step = Literal["post", "label", "stats"]


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
        text_column: Optional[str] = None,
        ollama_model: str = "llama3.2:3b-instruct-fp16",
        label_clusters: bool = True,
        generate_stats: bool = True,
        generate_visuals: bool = True,  # kept for backwards-compatibility
        process_parents: bool = True,
        skip_completed: bool = True,
        label_criteria=None,
        label_info=None,
        number_of_labels: int = 5,
        # NEW: choose exact subset of steps to run; None keeps legacy behavior
        steps: Optional[Sequence[Step]] = None,
    ):
        """
        Run any subset of the pipeline while preserving order:

            'post'  → post_process_hnmfk
            'label' → _label_all_clusters      (requires 'post' artifacts)
            'stats' → generate_cluster_stats    (requires 'post' artifacts)

        Rules:
          • 'label' and/or 'stats' can be run without 'post' only if artifacts already exist.
          • Order is always post → label → stats, even if you request multiple.
        """
        text_column = text_column or self.clean_cols_name

        # ---- resolve which steps to run, validate names ----
        if steps is not None:
            steps_set: Set[Step] = set(steps)
            invalid = steps_set.difference({"post", "label", "stats"})
            if invalid:
                raise ValueError(f"Invalid steps: {sorted(invalid)}; allowed: 'post','label','stats'")
            do_post = "post" in steps_set
            do_label = "label" in steps_set
            do_stats = "stats" in steps_set
        else:
            # Back-compat defaults using the existing booleans
            do_post = True
            do_label = bool(label_clusters)
            do_stats = bool(generate_stats)

        # ---- helper: ensure post-processing outputs exist if required ----
        def _assert_postprocessed_ready() -> None:
            missing = []
            for node in self.model.traverse_nodes():
                if node["leaf"] or process_parents:
                    w = node.get('W')
                    if w is None:
                        sig = node.get('signature')
                        if sig is None:
                            missing.append(f"{node.get('node_save_path', '<unknown>')} (no W or signature)")
                            continue
                        w = sig.reshape(-1, 1)

                    k = w.shape[1]
                    node_dir = Path(node["node_save_path"]).resolve().parent
                    cluster_file = node_dir / f"cluster_for_k={k}.csv"
                    top_words_file = node_dir / "top_words.csv"
                    if not (cluster_file.exists() and top_words_file.exists()):
                        missing.append(str(node_dir))

            if missing:
                raise RuntimeError(
                    "Labeling and/or stats require post-processing artifacts that were not found:\n"
                    + "\n".join(f"  - {m}" for m in missing)
                    + "\nInclude 'post' in `steps`, or run the post-processing step first."
                )

        # ─────────────────────────────────────────────────────────────
        # Step 1: POST-PROCESS (optional)
        # ─────────────────────────────────────────────────────────────
        if do_post:
            print("Step 1: Post-processing W/H matrix and cluster data...")
            self.postprocessor.post_process_hnmfk(
                hnmfk_model=self.model,
                V=vocab,
                D=data_df,
                col_name=text_column,
                skip_completed=skip_completed,
                process_parents=process_parents
            )
        else:
            if do_label or do_stats:
                _assert_postprocessed_ready()

        # ─────────────────────────────────────────────────────────────
        # Step 2: LABEL (optional; never before post)
        # ─────────────────────────────────────────────────────────────
        if do_label:
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

        # ─────────────────────────────────────────────────────────────
        # Step 3: STATS (optional; always last)
        # ─────────────────────────────────────────────────────────────
        if do_stats:
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
                w = node.get('W')
                if w is None:
                    sig = node.get('signature')
                    if sig is None:
                        continue
                    w = sig.reshape(-1, 1)

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

    # Convenience one-offs (unchanged)
    def run_labeling(self, df, top_words_df, ollama_model_name, label_criteria=None, additional_info=None, number_of_labels=5):
        return self.labeler.label_clusters_ollama(
            top_words_df=top_words_df,
            ollama_model_name=ollama_model_name,
            embedding_model=self.labeler.embedding_model,
            df=df,
            criteria=label_criteria,
            additional_information=additional_info,
            number_of_labels=number_of_labels
        )

    def run_postprocessing(self, V, D, col_name=None, **kwargs):
        return self.postprocessor.post_process_hnmfk(
            hnmfk_model=self.model,
            V=V,
            D=D,
            col_name=col_name or self.clean_cols_name,
            **kwargs
        )

    def run_stats(self, process_parents=True, skip_completed=True):
        return self.stats_generator.generate_cluster_stats(
            model=self.model,
            process_parents=process_parents,
            skip_completed=skip_completed
        )
