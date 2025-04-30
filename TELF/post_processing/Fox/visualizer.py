# fox/visualizer.py

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from .post_process_stats import (
    get_cluster_stat_file, create_tensor, process_affiliations_coaffiliations)
from ..Peacock.Plot.plot import plot_heatmap, plot_pie
from ..Peacock.Utility.aggregate_papers import aggregate_ostats, count_affiliations, count_countries
from ..Peacock.Utility.util import filter_by
from ..Wolf.plots import plot_matrix_graph
from ...helpers.data_structures import sum_dicts
from ...helpers.maps import get_id_to_name

class VisualizationManager:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def generate_all_statistics(self, processing_path: str, clean_cols_name: str = "clean_title_abstract"):
        df = pd.read_csv(processing_path)
        base_dir = Path(processing_path).parent

        try:
            summaries_df = pd.read_csv(base_dir / 'cluster_summaries.csv')
            summaries_df.label = summaries_df.label.astype(str)
        except FileNotFoundError:
            summaries_df = None

        top_words_df = pd.read_csv(base_dir / 'top_words.csv')
        stats_df = get_cluster_stat_file(df, top_words_df, summaries_df, n=5)
        stats_df.to_csv(base_dir / 'stats.csv', index=False)

        plot_df = self._prepare_plot_df(df)
        output_dir = base_dir

        col_names = self._get_col_names(plot_df)
        valid_clusters = [c for c in plot_df.cluster.unique() if pd.notna(c)]

        for c in tqdm(sorted(valid_clusters)):
            peacock_dir = os.path.join(output_dir, f'{int(c)}', 'peacock')
            os.makedirs(peacock_dir, exist_ok=True)
            self._generate_cluster_plots(plot_df, c, peacock_dir, col_names, clean_cols_name)

        self._generate_tensor_graphs(df, output_dir, 'author_ids', 'authors', 'co-authors')
        self._generate_tensor_graphs(df, output_dir, 'affiliations', 'affiliation_names', 'co-affiliations', is_affiliation=True)

    def _prepare_plot_df(self, df):
        if 'slic_authors' in df:
            return df.dropna(subset=['slic_authors', 'slic_author_ids', 'slic_affiliations']).copy()
        df = df.dropna(subset=['eid', 'authors', 'author_ids', 'affiliations']).drop_duplicates(subset=['eid'])
        df = df.reset_index(drop=True)
        df['id'] = df.index
        return df

    def _get_col_names(self, df):
        return {
            'id': 'id',
            'authors': 'slic_authors' if 'slic_authors' in df else 'authors',
            'author_ids': 'slic_author_ids' if 'slic_author_ids' in df else 'author_ids',
            'affiliations': 'slic_affiliations' if 'slic_affiliations' in df else 'affiliations',
            'funding': 'funding',
            'citations': 'num_citations',
            'references': 'references',
        }

    def _generate_cluster_plots(self, df, cluster_id, peacock_dir, col_names, clean_col):
        cluster_data = filter_by(df.copy(), filters={'cluster': cluster_id})
        author_stats = aggregate_ostats(cluster_data, key='author_id', sort_by='num_citations', top_n=100, col_names=col_names, by_year=False)
        author_stats.to_csv(os.path.join(peacock_dir, f'{int(cluster_id)}_author.csv'), index=False)

        affil_stats = aggregate_ostats(cluster_data, key='affiliation_id', sort_by='num_citations', top_n=100, col_names=col_names, by_year=False)
        affil_stats.to_csv(os.path.join(peacock_dir, f'{int(cluster_id)}_affiliation.csv'), index=False)

        for stat_col in ['num_citations', 'paper_count']:
            label = ' '.join([s.title() for s in stat_col.split('_')])

            # Author heatmap
            try:
                heatmap = aggregate_ostats(cluster_data, key='author_id', sort_by=stat_col, top_n=100, col_names=col_names, by_year=True)
                heatmap = heatmap.drop_duplicates(subset=['year', stat_col]).dropna(subset=['year', stat_col])
                heatmap = heatmap.pivot(index='author_id', columns='year', values=stat_col)

                labels = {row.author_id: {'Name': row.author, 'Affiliation': row.affiliation} for _, row in author_stats.iterrows()}
                plot_heatmap(heatmap.T, cmap='jet_white', fname=os.path.join(peacock_dir, f'{int(cluster_id)}_{stat_col}_author_heatmap.html'),
                             labels=labels, title=f'{label} Author Heatmap for Cluster {cluster_id}', xlabel='Author ID', ylabel='Year')
            except Exception as e:
                print(f"[Cluster {cluster_id}] Author heatmap error: {e}")

            # Affiliation heatmap
            try:
                heatmap = aggregate_ostats(cluster_data, key='affiliation_id', sort_by=stat_col, top_n=100, col_names=col_names, by_year=True)
                heatmap = heatmap.drop_duplicates(subset=['year', stat_col]).dropna(subset=['year', stat_col])
                heatmap = heatmap.pivot(index='affiliation_id', columns='year', values=stat_col)

                labels = {row.affiliation_id: {'Affiliation Name': row.affiliation} for _, row in affil_stats.iterrows()}
                plot_heatmap(heatmap.T, cmap='jet_white', fname=os.path.join(peacock_dir, f'{int(cluster_id)}_{stat_col}_affiliation_heatmap.html'),
                             labels=labels, title=f'{label} Affiliation Heatmap for Cluster {cluster_id}', xlabel='Affiliation ID', ylabel='Year')
            except Exception as e:
                print(f"[Cluster {cluster_id}] Affiliation heatmap error: {e}")

        # Pie chart inputs
        tokens = [x.split() for x in cluster_data[clean_col].dropna()]
        word_counts = sum_dicts([dict(Counter(x)) for x in tokens], n=50)
        country_counts = count_countries(cluster_data)
        affiliation_counts = count_affiliations(cluster_data)

        try:
            plot_pie(affiliation_counts, words=word_counts, fname=os.path.join(peacock_dir, f'{int(cluster_id)}_affiliation_pie.png'))
        except Exception as e:
            print(f"[Cluster {cluster_id}] Pie affiliation error: {e}")

        try:
            plot_pie(country_counts, words=word_counts, fname=os.path.join(peacock_dir, f'{int(cluster_id)}_country_pie.png'))
        except Exception as e:
            print(f"[Cluster {cluster_id}] Pie country error: {e}")

    def _generate_tensor_graphs(self, df, output_dir, id_col, name_col, plot_suffix, is_affiliation=False):
        wolf_df = df.dropna(subset=[id_col]).copy().reset_index(drop=True)

        if is_affiliation:
            wolf_df['affiliation_ids'], wolf_df['affiliation_names'] = zip(*wolf_df['affiliations'].apply(process_affiliations_coaffiliations))

        name_map = get_id_to_name(wolf_df, name_col, id_col)
        attributes = {k: {'name': v} for k, v in name_map.items()}

        for c in tqdm(sorted(wolf_df.cluster.dropna().unique())):
            cluster_df = wolf_df.loc[wolf_df.cluster == c].reset_index(drop=True)
            if cluster_df.empty or cluster_df[id_col].isna().all():
                continue

            try:
                X, node_map, time_map = create_tensor(cluster_df, id_col, n_jobs=-1, verbose=False)
            except ValueError as e:
                print(f"[Cluster {c}] Skipping {plot_suffix} tensor due to error: {e}")
                continue

            fig = plot_matrix_graph(X.sum(axis=2), node_map, node_attributes=attributes, filter_isolated_nodes=True, verbose=False)
            peacock_dir = os.path.join(output_dir, f'{int(c)}', 'peacock')
            fig.write_html(os.path.join(peacock_dir, f'{int(c)}_{plot_suffix}.html'))
