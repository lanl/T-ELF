# fox/clustering_analyzer.py

import os
import shutil
import numpy as np
import pandas as pd

from .post_process_functions import (H_cluster_argmax, get_core_map,
                                     best_n_papers, sme_attribution)
from ...helpers.stats import top_words
from ...helpers.figures import word_cloud
from ...helpers.file_system import process_terms, check_path
from ...helpers.figures import plot_H_clustering
from ...helpers.stats import H_clustering
from ...pre_processing.Vulture.tokens_analysis.top_words import get_top_words

class ClusteringAnalyzer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def analyze_clusters(self, H, W, vocab, df, output_dir, archive_dir,
                         top_words_per_cluster, clean_cols_name, terms):
        labels, counts, table_df = H_cluster_argmax(H)
        table_df.to_csv(os.path.join(output_dir, 'table_H-clustering.csv'), index=False)
        np.savetxt(os.path.join(archive_dir, 'cluster_documents_map.txt'), np.rint(labels), fmt='%d')

        words, probabilities = top_words(W, vocab, top_words_per_cluster)
        words_df = pd.DataFrame(words)
        probabilities_df = pd.DataFrame(probabilities)

        words_df.to_csv(os.path.join(output_dir, 'top_words.csv'), index=False)
        probabilities_df.to_csv(os.path.join(output_dir, 'probabilities_top_words.csv'), index=False)

        word_cloud(words, probabilities=probabilities, path=output_dir, max_words=top_words_per_cluster,
                   mask=np.zeros((800, 800)), max_font_size=80, contour_width=1)


        clusters_information, documents_information = H_clustering(H, verbose=True)
        plot_H_clustering(H, name=os.path.join(output_dir, "centroids_H_clustering"))

        clusters_information_df = pd.DataFrame(clusters_information).T
        documents_information_df = pd.DataFrame(documents_information).T

        clusters_information_df.to_csv(os.path.join(archive_dir,'clusters_information.csv'), index=False)
        documents_information_df.to_csv(os.path.join(archive_dir,'documents_information.csv'), index=False)

        if 'year' in df:
            df['year'] = df['year'].fillna(-1).astype(int).replace(-1, np.nan)

        df['cluster'] = documents_information_df['cluster']
        for col in ['cluster_coordinates', 'similarity_to_cluster_centroid']:
            if col in documents_information_df.columns:
                df[col] = documents_information_df[col]
            else:
                df[col] = None


        if 'type' in df:
            core_map = get_core_map(df)
            num_total_core = len(df.loc[df.type == 0])
            table_df['core_count'] = table_df['cluster'].map(core_map)
            if num_total_core > 0:
                core_map_counts = {k: round(100*v / num_total_core, 2) for k,v in core_map.items()}
                table_df['core_percentage'] = table_df['cluster'].map(core_map_counts)
                table_df.to_csv(os.path.join(output_dir, 'table_H-clustering.csv'), index=False)

        post_processed_df = f'cluster_for_k={len(set(labels))}.csv'
        post_processed_df_path = os.path.join(output_dir, post_processed_df)
        df.to_csv(post_processed_df_path, index=False)

        best_n_papers(df, output_dir, top_words_per_cluster)

        if terms is not None:
            _, highlighting_map = process_terms(terms)
            terms = list(highlighting_map.keys())
            sme_attribution(df, output_dir, terms, col=clean_cols_name)

        for cluster_id in sorted(df['cluster'].dropna().unique()):
            save_dir = check_path(os.path.join(output_dir, f'{int(cluster_id)}'))
            cluster_df = df[df['cluster'] == cluster_id]
            clean_documents = cluster_df[clean_cols_name].to_dict()

            top_1grams = get_top_words(clean_documents, top_n=100, n_gram=1, verbose=True)
            top_1grams.to_csv(os.path.join(save_dir, f'{int(cluster_id)}_bow_unigrams.csv'), index=False)

            top_2grams = get_top_words(clean_documents, top_n=100, n_gram=2, verbose=True)
            top_2grams.to_csv(os.path.join(save_dir, f'{int(cluster_id)}_bow_bigrams.csv'), index=False)

        for cluster_id in sorted(df['cluster'].dropna().unique()):

            # Move the generated cluster plots to their corresponding folders
            src = os.path.join(output_dir, f'centroids_H_clustering_{int(cluster_id)}.png')
            dst_dir = os.path.join(output_dir, f'{int(cluster_id)}')
            dst = os.path.join(dst_dir, f'centroids_H_clustering_{int(cluster_id)}.png')

            if os.path.exists(src):
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src, dst)
                if self.verbose:
                    print(f"[OK] Moved {src} → {dst}")
            else:
                print(f"[WARN] Expected image not found: {src}")


        return post_processed_df_path
