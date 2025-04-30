import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from matplotlib.ticker import MultipleLocator

class ModelPerformancePlotter:
    def __init__(
        self,
        data,  # Either a list of CSV paths or a pd.DataFrame
        baseline_model="BASELINE",
        baseline_score=1.0,
        metrics=None,
        model_name_map=None,
        color_map=None,
        hatch_map=None,
    ):
        self.data_input = data
        self.baseline_model = baseline_model
        self.baseline_score = baseline_score
        self.metrics = metrics or ['factcc_score','nli_entailment','rougeL','summaC_score']
        self.name_map = model_name_map or {}
        self.color_map = color_map or {}
        self.hatch_map = hatch_map or {}

    def load_and_prepare(self):
        # Load DataFrame from paths or use provided DataFrame
        if isinstance(self.data_input, pd.DataFrame):
            df = self.data_input.copy()
        elif isinstance(self.data_input, list):
            dfs = [pd.read_csv(p) for p in self.data_input]
            df = pd.concat(dfs, axis=0)
        else:
            raise ValueError("data must be a DataFrame or list of CSV file paths")

        # Fix baseline
        df.loc[df['model'] == self.baseline_model, 'factcc_score'] = self.baseline_score

        # Rename models
        df['model'] = df['model'].map(self.name_map).fillna(df['model'])

        # Melt and average
        df_long = df.melt(
            id_vars='model',
            value_vars=self.metrics,
            var_name='Metric',
            value_name='Score'
        )
        df_avg = df_long.groupby(['Metric', 'model'])['Score'].mean().reset_index()
        df_avg['Metric'] = pd.Categorical(df_avg['Metric'], categories=self.metrics, ordered=True)

        self.df_avg = df_avg.sort_values(['Metric', 'model'])
        self.models = sorted(self.df_avg['model'].unique())

    def plot(self, save_path="model_performance.pdf", figsize=(15,5)):
        self.load_and_prepare()

        sns.set(style='whitegrid', font_scale=1.9)
        fig, ax = plt.subplots(figsize=figsize)

        barplot = sns.barplot(
            data=self.df_avg,
            x='Metric', y='Score', hue='model',
            order=self.metrics, hue_order=self.models,
            palette=self.color_map, ax=ax
        )

        # Apply hatches
        for container, model in zip(ax.containers, self.models):
            for bar in container:
                bar.set_hatch(self.hatch_map.get(model, ""))

        # y-ticks and legend
        fig.canvas.draw()
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

        handles = []
        for container, model in zip(ax.containers, self.models):
            bar = container[0]
            handles.append(
                mpatches.Patch(
                    facecolor=bar.get_facecolor(),
                    edgecolor=bar.get_edgecolor(),
                    hatch=bar.get_hatch(),
                    label=model,
                    linewidth=bar.get_linewidth()
                )
            )

        ax.legend(
            handles=handles,
            title='Model',
            loc='upper right',
            fontsize=16,
            title_fontsize=14,
            ncol=2
        )

        ax.set_xticks(np.arange(len(self.metrics)))
        ax.set_xticklabels(self.metrics, rotation=0, ha='center')
        ax.set_ylabel('Average Score')
        ax.set_xlabel('')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
