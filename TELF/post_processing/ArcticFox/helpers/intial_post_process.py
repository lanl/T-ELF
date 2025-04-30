from pathlib import Path
import os
import numpy as np
import pandas as pd

from ....helpers.figures import plot_H_clustering
from ....helpers.stats import H_clustering
from ....helpers.figures import create_wordcloud 
from ....pre_processing.Vulture.tokens_analysis.top_words import get_top_words
from ...Fox.post_process_functions import (H_cluster_argmax, get_core_map, best_n_papers)
from ....helpers.file_system import check_path
from ....helpers.stats import top_words

class HNMFkPostProcessor:
    def __init__(
        self,
        top_n_words=50,
        out_dir='./post_result',
        archive_subdir='archive',
        table_filename='table_H-clustering.csv',
        cluster_doc_map_filename='cluster_documents_map.txt',
        top_words_filename='top_words.csv',
        probs_filename='probabilities_top_words.csv',
        clusters_info_filename='clusters_information.csv',
        documents_info_filename='documents_information.csv',
        default_clean_col='clan_title_abstract',
        wordcloud_size=(800, 800),
        max_font_size=80,
        contour_width=1,
        clustering_distance='max',

        # Column names
        col_year='year',
        col_type='type',
        col_cluster='cluster',
        col_cluster_coords='cluster_coordinates',
        col_similarity='similarity_to_cluster_centroid',
    ):
        self.top_n_words = top_n_words
        self.out_dir = out_dir
        self.archive_subdir = archive_subdir
        self.table_filename = table_filename
        self.cluster_doc_map_filename = cluster_doc_map_filename
        self.top_words_filename = top_words_filename
        self.probs_filename = probs_filename
        self.clusters_info_filename = clusters_info_filename
        self.documents_info_filename = documents_info_filename
        self.default_clean_col = default_clean_col
        self.wordcloud_size = wordcloud_size
        self.max_font_size = max_font_size
        self.contour_width = contour_width
        self.clustering_distance = clustering_distance

        # Column name configs
        self.col_year = col_year
        self.col_type = col_type
        self.col_cluster = col_cluster
        self.col_cluster_coords = col_cluster_coords
        self.col_similarity = col_similarity

    def handle_input_variable(self, var):
        if isinstance(var, str) and var.endswith('.txt'):
            with open(var) as f:
                return [l.strip() for l in f]
        elif isinstance(var, str) and var.endswith('.csv'):
            return pd.read_csv(var)
        return var

    def modded_post_process(self, vocab, W, H, df_post, clean_cols_name=None, attribute_columns=None, out_dir=None):
        out_dir = out_dir or self.out_dir
        clean_cols_name = clean_cols_name or self.default_clean_col
        archive_dir = os.path.join(out_dir, self.archive_subdir)
        os.makedirs(archive_dir, exist_ok=True)

        # 1. H clustering
        labels, counts, table_df = H_cluster_argmax(H)
        labels = np.array([labels]) if np.isscalar(labels) else labels
        if labels.ndim == 0:
            labels = labels.reshape(1)

        # 2. Save overall cluster info
        table_df_path = os.path.join(out_dir, self.table_filename)
        table_df.to_csv(table_df_path, index=False)
        np.savetxt(os.path.join(archive_dir, self.cluster_doc_map_filename), np.rint(labels), fmt='%d')

        # 3. Top words for whole matrix
        vocab = np.array(vocab)
        print('from modded_post_process, len(W):', len(W), ", len(vocab):", len(vocab))
        words, probabilities = top_words(W, vocab, self.top_n_words)
        pd.DataFrame(words).to_csv(os.path.join(out_dir, self.top_words_filename), index=False)
        pd.DataFrame(probabilities).to_csv(os.path.join(out_dir, self.probs_filename), index=False)

        # 4. Create general wordcloud and H clustering plot
        create_wordcloud(
            W=W,
            vocab=vocab,
            top_n=self.top_n_words,
            path=out_dir,
            verbose=False,
            max_words=self.top_n_words,
            mask=np.zeros(self.wordcloud_size),
            background_color="black",
            max_font_size=self.max_font_size,
            contour_width=self.contour_width,
            grid_dimension=4  
        )

        clusters_info, docs_info = H_clustering(H, verbose=True)
        # plot_H_clustering(H, name=os.path.join(out_dir, "centroids_H_clustering"))

        # 5. Save general documents and cluster info
        pd.DataFrame(clusters_info).T.to_csv(os.path.join(archive_dir, self.clusters_info_filename), index=False)
        documents_information_df = pd.DataFrame(docs_info).T
        documents_information_df.to_csv(os.path.join(archive_dir, self.documents_info_filename), index=False)

        # 6. Update df_post with clustering results
        if self.col_year in df_post:
            df_post[self.col_year] = df_post[self.col_year].fillna(-1).astype(int).replace(-1, np.nan)

        df_post[self.col_cluster] = documents_information_df['cluster']
        if 'cluster_coordinates' in documents_information_df.columns:
            df_post[self.col_cluster_coords] = documents_information_df['cluster_coordinates']
        else:
            print("Warning: 'cluster_coordinates' column missing. Skipping assignment.")

        df_post[self.col_similarity] = documents_information_df['similarity_to_cluster_centroid']

        # 7. If type column exists, calculate core stats
        if self.col_type in df_post:
            core_map = get_core_map(df_post)
            num_total_core = len(df_post.loc[df_post[self.col_type] == 0])
            if num_total_core > 0:
                table_df = table_df.assign(
                    core_count=table_df['cluster'].map(core_map),
                    core_percentage=table_df['cluster'].map(lambda c: round(100 * core_map.get(c, 0) / num_total_core, 2))
                )
                table_df.to_csv(table_df_path, index=False)

        # 8. Save final post-processed dataframe
        post_processed_df_path = os.path.join(out_dir, f'cluster_for_k={W.shape[1]}.csv')
        df_post.to_csv(post_processed_df_path, index=False)

        # 9. Save best_n_papers output
        best_n_papers(df_post, out_dir, self.top_n_words)

        # 10. 🔥 For each cluster, save its own outputs
        for cluster_id in sorted(df_post[self.col_cluster].unique()):
            cluster_id = int(cluster_id)
            save_dir = check_path(os.path.join(out_dir, f'{cluster_id}'))
            cluster_df = df_post.loc[df_post[self.col_cluster] == cluster_id].copy()
            clean_documents = cluster_df[clean_cols_name].to_dict()

            # (a) save BOW for each cluster
            for n in [1, 2]:
                bow_df = get_top_words(clean_documents, top_n=100, n_gram=n, verbose=True)
                bow_df.to_csv(os.path.join(save_dir, f'{cluster_id}_bow_{["unigrams", "bigrams"][n-1]}.csv'), index=False)

            # (b) save wordcloud per cluster
            W_cluster = W[:, cluster_id].reshape(-1, 1)  # W[:, i] (as 2D matrix)
            create_wordcloud(
                W=W_cluster,
                vocab=vocab,
                top_n=self.top_n_words,
                path=save_dir,
                verbose=False,
                max_words=self.top_n_words,
                mask=np.zeros(self.wordcloud_size),
                background_color="black",
                max_font_size=self.max_font_size,
                contour_width=self.contour_width,
                filename_base=cluster_id
            )
            # Your create_wordcloud will automatically save 0.pdf into that cluster dir

            # (c) save H clustering centroids per cluster (optional)
            H_cluster = H[cluster_id, :].reshape(1, -1)
            plot_H_clustering(
                H_cluster,
                name=os.path.join(save_dir, f"centroids_H_clustering_{cluster_id}.png")
            )

        return post_processed_df_path

    def post_process_hnmfk(self, hnmfk_model, V, D, col_name=None, attributes=None, skip_completed=True, process_parents=False):
        vocab = self.handle_input_variable(V)
        data = self.handle_input_variable(D)

        all_nodes = hnmfk_model.traverse_nodes()
        for node in all_nodes:
            if node["leaf"] or process_parents:
                w, h = node.get('W'), node.get('H')
                if w is None and h is None:
                    w = node['signature'].reshape(-1, 1)
                    h = node['probabilities'].reshape(1, -1)

                node_post_process_dir = Path(node['node_save_path']).resolve().parent
                cluster_data = f'cluster_for_k={w.shape[1]}.csv'

                if skip_completed and (node_post_process_dir / cluster_data).exists():
                    continue

                node_indices = list(node["original_indices"])
                node_df = data.iloc[node_indices].reset_index(drop=True)

                print('form inital_post_process.py, in post_process_hnmfk len(W)',len(w),", len(vocab):",len(vocab))
                print("Dir:",node_post_process_dir)

                self.modded_post_process(
                    vocab=vocab,
                    W=w,
                    H=h,
                    df_post=node_df,
                    clean_cols_name=col_name,
                    attribute_columns=attributes,
                    out_dir=node_post_process_dir
                )
