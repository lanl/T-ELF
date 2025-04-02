import os
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import Counter

from ...Peacock.Utility.aggregate_papers import (aggregate_ostats, count_countries,
                                                 count_affiliations)
from ...Peacock.Plot.plot import (plot_heatmap, plot_bar, 
                                  plot_scatter, plot_pie)

from ...Fox.post_process_stats import (get_cluster_stat_file, sum_dicts)

class HNMFkStatsGenerator:
    def __init__(self, clean_cols_name="clean_title_abstract"):
        self.clean_cols_name = clean_cols_name

    def check_missing_peacock_dir(self, base_path):
        missing_peacock_dirs = []
        base_path = Path(base_path)
        for root, dirs, _ in os.walk(base_path):
            dirs[:] = [d for d in dirs if d not in {'archive', 'peacock'} and 'NMFk' not in d and 'ipynb_checkpoints' not in d]
            for d in dirs:
                subdir_path = Path(root) / d
                if not (subdir_path / 'peacock').exists():
                    missing_peacock_dirs.append(subdir_path)
        return missing_peacock_dirs

    def replace_and_remove_source(self, source_dir, output_dir):
        source_path = Path(source_dir)
        destination_path = Path(output_dir)

        if destination_path.exists():
            source_peacock = source_path / 'peacock'
            dest_peacock = destination_path / 'peacock'
            if source_peacock.exists():
                if dest_peacock.exists():
                    shutil.rmtree(dest_peacock)
                shutil.move(str(source_peacock), str(dest_peacock))
            shutil.rmtree(source_path)
        else:
            shutil.move(str(source_path), str(destination_path))

    def generate_cluster_stats(self, model, process_parents=True, skip_completed=True):
        all_nodes = model.traverse_nodes()
        completed = 0

        for node in all_nodes:
            if node["leaf"] or process_parents:
                w = node.get('W')
                if w is None:
                    w = node['signature'].reshape(-1, 1)

                node_dir = Path(node['node_save_path']).resolve().parent
                cluster_data_path = node_dir / f'cluster_for_k={w.shape[1]}.csv'
                needs_stats = self.check_missing_peacock_dir(node_dir)

                completed += 1
                if needs_stats or not skip_completed:
                    print(f"Processing node: {node_dir}")

                    df = pd.read_csv(cluster_data_path)
                    output_dir = node_dir

                    try:
                        summaries_df = pd.read_csv(output_dir / 'cluster_summaries.csv')
                        summaries_df.label = summaries_df.label.astype(str)
                    except FileNotFoundError:
                        summaries_df = None

                    top_words_df = pd.read_csv(output_dir / 'top_words.csv')
                    try:

                        stats_df = get_cluster_stat_file(df, top_words_df, summaries_df, n=5)
                        stats_df.to_csv(output_dir / 'stats.csv', index=False)

                        self._generate_peacock_stats(df, output_dir, stats_df)
                    except KeyError as e:
                        print(f"Skipping stats generation for {node_dir} due to missing column: {e}")
                    except Exception as e:
                        print(f"Unexpected error generating stats for {node_dir}: {e}")

        print(f"\n{completed} out of {len(all_nodes)} nodes processed.")

    def _generate_peacock_stats(self, df, output_dir, stats_df):
        print("Generating Peacock stats...")

        col_names = {
            'id': 'id',
            'authors': next((col for col in ['slic_authors', 'authors'] if col in df.columns), None),
            'author_ids': next((col for col in ['slic_author_ids', 'author_ids'] if col in df.columns), None),
            'affiliations': next((col for col in ['slic_affiliations', 'affiliations'] if col in df.columns), None),
            'funding': 'funding' if 'funding' in df.columns else None,
            'citations': 'num_citations' if 'num_citations' in df.columns else None,
            'references': 'references' if 'references' in df.columns else None,
        }
        col_names = {k: v for k, v in col_names.items() if v is not None}

        required_cols = [col_names.get(c) for c in ['authors', 'author_ids', 'affiliations'] if col_names.get(c)]
        if required_cols:
            plot_df = df.dropna(subset=required_cols).drop_duplicates(subset=['eid'] if 'eid' in df.columns else None)
        else:
            plot_df = df.copy()

        plot_df = plot_df.reset_index(drop=True)
        plot_df['id'] = plot_df.index

        for cluster in tqdm(sorted(plot_df.cluster.unique())):
            cluster_dir = output_dir / f'{int(cluster)}' / 'peacock'
            cluster_dir.mkdir(parents=True, exist_ok=True)
            cluster_data = plot_df[plot_df.cluster == cluster]

            self._generate_affiliation_author_stats(cluster_data, cluster_dir, cluster, col_names)
            self._generate_all_visuals(cluster_data, cluster_dir, cluster, col_names, stats_df)

    def _generate_affiliation_author_stats(self, df, peacock_dir, cluster, col_names):
        if 'author_ids' in col_names:
            author_df = aggregate_ostats(df, key='author_id', sort_by='num_citations', top_n=100, col_names=col_names, by_year=False)
            author_df.to_csv(peacock_dir / f'{int(cluster)}_author.csv', index=False)

        if 'affiliations' in col_names:
            aff_df = aggregate_ostats(df, key='affiliation_id', sort_by='num_citations', top_n=100, col_names=col_names, by_year=False)
            aff_df.to_csv(peacock_dir / f'{int(cluster)}_affiliation.csv', index=False)

    def _generate_all_visuals(self, cluster_df, peacock_dir, cluster, col_names, stats_df):
        try:
            self._plot_heatmaps(cluster_df, peacock_dir, cluster, col_names)
            self._plot_bar_charts(cluster_df, peacock_dir, cluster, col_names)
            self._plot_scatter(cluster_df, peacock_dir, cluster, col_names)
            self._plot_pies(cluster_df, peacock_dir, cluster)
            self._rename_cluster_folder(peacock_dir.parent, cluster, stats_df)
        except Exception as e:
            print(f"Error plotting visuals for cluster {cluster}: {e}")

    def _plot_heatmaps(self, df, peacock_dir, cluster, col_names):
        for stat_col in ['num_citations', 'paper_count']:
            if stat_col not in df.columns:
                continue
            label = ' '.join([s.title() for s in stat_col.split('_')])
            for key, label_key in [('author_id', 'author'), ('affiliation_id', 'affiliation')]:
                if key not in df.columns:
                    continue
                data = aggregate_ostats(df, key=key, sort_by=stat_col, top_n=100, col_names=col_names, by_year=True)
                data = data.drop_duplicates(subset=['year', stat_col]).dropna(subset=['year', stat_col])
                heatmap = data.pivot(index=key, columns='year', values=stat_col)

                labels_map = {
                    row[key]: {
                        'Name': row.get('author', ''),
                        'Affiliation': row.get('affiliation', '')
                    }
                    for _, row in data.iterrows()
                }

                plot_heatmap(
                    heatmap.T,
                    cmap='jet_white',
                    fname=peacock_dir / f'{cluster}_{stat_col}_{label_key}_heatmap.html',
                    labels=labels_map,
                    title=f'{label} {label_key.title()} Heatmap for Cluster {cluster}',
                    xlabel=f'{label_key.title()} ID',
                    ylabel='Year'
                )

    def _plot_bar_charts(self, df, peacock_dir, cluster, col_names):
        for key in ['affiliation_id', 'author_id']:
            if key not in df.columns:
                continue
            label_key = 'affiliation' if key == 'affiliation_id' else 'author'
            stat_cols = ['num_citations', 'paper_count', 'attribution_percentage']
            data = aggregate_ostats(df, key=key, sort_by='num_citations', top_n=10, col_names=col_names, by_year=False)

            plot_bar(
                data=data,
                annotate=True,
                colorbar=True,
                x=key,
                ys=stat_cols,
                labels=[label_key, 'country'] if key == 'affiliation_id' else [label_key, 'affiliation', 'country'],
                title=f'{label_key.title()} Bar Plot for Cluster {cluster}',
                xlabel=f'{label_key.title()} ID',
                ylabel='Num Citations',
                cmap='temps',
                fname=peacock_dir / f'{cluster}_{label_key}_bar.html'
            )

    def _plot_scatter(self, df, peacock_dir, cluster, col_names):
        stat_cols = ['num_citations', 'paper_count', 'attribution_percentage']
        for key in ['author_id', 'affiliation_id']:
            if key not in df.columns:
                continue
            label_key = 'author' if key == 'author_id' else 'affiliation'
            stats_df = aggregate_ostats(df, key=key, sort_by='num_citations', top_n=100, col_names=col_names, by_year=False)

            plot_scatter(
                stats_df,
                hue='country' if key == 'affiliation_id' else 'affiliation',
                labels=[key, label_key],
                x=stat_cols[0], xlabel='Num Citations',
                y=stat_cols[1], ylabel='Paper Count',
                z=stat_cols[2], zlabel='Attribution %',
                n=10,
                interactive=True,
                annotate=True,
                width=12,
                height=6,
                labels_add_scatter_cols=True,
                fname=peacock_dir / f'{cluster}_{label_key}_scatter.html'
            )

    def _plot_pies(self, df, peacock_dir, cluster):
        tokens = [x.split() for x in df[self.clean_cols_name].dropna()] if self.clean_cols_name in df else []
        word_counts = sum_dicts([dict(Counter(x)) for x in tokens], n=50)
        top_countries = count_countries(df)
        top_affiliations = count_affiliations(df)

        plot_pie(top_affiliations, words=word_counts, fname=peacock_dir / f'{cluster}_affiliation_pie.png')
        plot_pie(top_countries, words=word_counts, fname=peacock_dir / f'{cluster}_country_pie.png')

    def _rename_cluster_folder(self, output_dir, cluster, stats_df):
        try:
            row = stats_df.loc[stats_df['cluster'] == cluster].iloc[0]
            label = row['label'][:30].replace(" ", "_") if row['label'] else ""
            new_name = f"{cluster}-{label}_{row['num_papers']}-documents" if label else f"{cluster}_{row['num_papers']}-documents"
            self.replace_and_remove_source(output_dir / str(cluster), output_dir / new_name)
        except Exception as e:
            print(f"Could not rename cluster folder {cluster}: {e}")
