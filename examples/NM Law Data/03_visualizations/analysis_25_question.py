import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

class AttemptAccuracyPlotter:
    def __init__(self, data, max_score_per_question=3):
        self.data = data
        self.max_score = max_score_per_question
        self.categories = list(data.keys())
        self.values_attempted = []
        self.values_accuracy = []
        self.colors = ["#A569BD", "#8B0000", "#74CA4D", "#36C9C6", "#4B0082"]
        self.patterns = ['/', '\\', '|', '-', '+']
    
    def calculate_values(self):
        for category, questions in self.data.items():
            total_questions = len(questions)
            attempted = sum(q["attempt"] for q in questions.values())
            accuracy_attempted = sum(q["accuracy"] for q in questions.values() if q["attempt"] == 1)

            attempted_percentage = (attempted / total_questions) * 100
            accuracy_percentage = (
                (accuracy_attempted / (attempted * self.max_score)) * 100
                if attempted > 0 else 0
            )

            self.values_attempted.append(attempted_percentage)
            self.values_accuracy.append(accuracy_percentage)

    def plot(self, save_path="attempted_accuracy_with_pattern_legend.pdf"):
        self.calculate_values()
        plt.rcParams.update({'font.size': 18})

        x_attempted = np.arange(len(self.categories))
        gap_between_groups = 0.2
        x_accuracy = x_attempted + len(self.categories) + gap_between_groups
        bar_width = 1

        fig, ax = plt.subplots(figsize=(14, 7))

        for i, cat in enumerate(self.categories):
            ax.bar(
                x_attempted[i], self.values_attempted[i],
                width=bar_width,
                color=self.colors[i % len(self.colors)],
                hatch=self.patterns[i % len(self.patterns)],
            )
            ax.bar(
                x_accuracy[i], self.values_accuracy[i],
                width=bar_width,
                color=self.colors[i % len(self.colors)],
                hatch=self.patterns[i % len(self.patterns)],
            )

        # X-axis labels
        x_ticks = np.concatenate([x_attempted, x_accuracy])
        x_labels = self.categories + self.categories
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=15, fontsize=16)

        # Section headers
        for section, pos in zip(["Attempted", "Accuracy"], [x_attempted.mean(), x_accuracy.mean()]):
            ax.text(pos, 97, section, ha='center', va='center', fontsize=30, weight='bold')

        ax.set_ylabel("Percentage (%)", fontsize=20)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Legend
        legend_handles = [
            Patch(
                facecolor=self.colors[i % len(self.colors)],
                hatch=self.patterns[i % len(self.patterns)],
                label=cat,
                edgecolor='black'
            )
            for i, cat in enumerate(self.categories)
        ]
        ax.legend(handles=legend_handles, title="Model", title_fontsize=18,
                  fontsize=16, loc='upper center', bbox_to_anchor=(0.48, -0.15), ncol=5)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
