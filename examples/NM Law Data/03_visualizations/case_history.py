import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe

class CourtCasesPlotter:
    def __init__(self,
                 supreme_court_data: dict,
                 appellate_data: dict,
                 sup_events: dict = None,
                 app_events: dict = None,
                 supreme_color: str = '#cc7000',
                 appeal_color: str = '#d4af00'):
        # --- force all keys to ints here ---
        self.supreme_data   = {int(year): val for year, val in supreme_court_data.items()}
        self.appellate_data = {int(year): val for year, val in appellate_data.items()}
        self.sup_events     = {int(year): lbl for year, lbl in (sup_events or {}).items()}
        self.app_events     = {int(year): lbl for year, lbl in (app_events or {}).items()}

        self.supreme_color = supreme_color
        self.appeal_color = appeal_color
        self._configure_style()

    def _configure_style(self):
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.figsize': (6, 2.5),
            'figure.dpi': 300,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.7,
        })

    def plot(self, save_path: str = None):
        # now these are real ints, so this is a numeric axis
        years    = sorted(set(self.supreme_data) | set(self.appellate_data))
        sup_cases = np.array([ self.supreme_data.get(y, np.nan) for y in years ], dtype=float)
        app_cases = np.array([ self.appellate_data.get(y, np.nan) for y in years ], dtype=float)

        fig, ax = plt.subplots()
        ax.plot(years, sup_cases,
                marker='o', markersize=5, markerfacecolor='lightgrey',
                markeredgewidth=1, color=self.supreme_color,
                linewidth=1.2, label='Supreme Court Cases')
        ax.plot(years, app_cases,
                marker='^', markersize=5, markerfacecolor='black',
                markeredgewidth=1, color=self.appeal_color,
                linewidth=1.2, label='Appeals Court Cases')

        # annotate your (now-numeric) event years
        y_max = ax.get_ylim()[1]
        x_offset = 0.5
        for yr, lbl in self.sup_events.items():
            ax.axvline(yr, color=self.supreme_color, linestyle='--', alpha=0.7)
            ax.annotate(lbl,
                        xy=(yr, y_max*0.9),
                        xytext=(yr + x_offset, y_max*0.7),
                        textcoords='data', rotation=90,
                        va='center', ha='left', fontsize=12,
                        color=self.supreme_color,
                        path_effects=[ pe.Stroke(linewidth=1.5, foreground='black', alpha=0.05),
                                       pe.Normal() ])
        for yr, lbl in self.app_events.items():
            ax.axvline(yr, color=self.appeal_color, linestyle='--', alpha=0.7)
            ax.annotate(lbl,
                        xy=(yr, y_max*0.9),
                        xytext=(yr + x_offset, y_max*0.7),
                        textcoords='data', rotation=90,
                        va='center', ha='left', fontsize=12,
                        color=self.appeal_color,
                        path_effects=[ pe.Stroke(linewidth=1.5, foreground='black', alpha=0.05),
                                       pe.Normal() ])

        # put decade ticks on a numeric axis
        start = (years[0] // 10) * 10
        end   = ((years[-1] // 10) + 1) * 10
        ticks = list(range(start, end + 1, 10))
        ax.set_xticks(ticks)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Cases')
        ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.16), frameon=False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.show()
