
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_interaction_grid(R, selected_communities=None, axes_font_size=24, ylim=None, 
              				ylim_diag=None, base_year=1995, save_path="", name=None):

	# if input is a range object, convert to list
	if not isinstance(selected_communities, list):
		communities = list(selected_communities)
	else:
		communities = selected_communities

	num_years = R.shape[0]
	num_communities = len(communities)

	# find the y limit
	if not ylim:
		ylim = (0, np.max([R[t][a, b] for t in range(num_years) 
					for a in communities 
					for b in communities]))
	if not ylim_diag:  # set y limits equal if diagonal not specified
		ylim_diag = ylim

	interactions = np.sum(R,axis=0) > 1e-3  # decide what interactions to plot

	fig = plt.figure(constrained_layout=True)
	fig.set_figheight(num_communities * 2)  # set the figure proportions in 3:2 ratio
	fig.set_figwidth(num_communities * 3)
	spec = gridspec.GridSpec(ncols=num_communities, nrows=num_communities, figure=fig)
	for i, a in tqdm(enumerate(communities)):
		for j, b in enumerate(communities):
			ax = fig.add_subplot(spec[i, j])
			if interactions[a,b]:
				x = [t + base_year for t in range(num_years)]
				y = [R[t][a,b] for t in range(num_years)]
				if i != j:
					ax.plot(x,y, lw=3, c='blue')               
					plt.setp(ax, ylim=ylim)  # fix the plots to use the same scale
				else:  # highlight the diagonal
					ax.plot(x,y, lw=3, c='red') 
					plt.setp(ax, ylim=ylim_diag)  # set custom scale for diagonals

				# set xticks to show every 5 years, starting with first year   
				plt.xticks(list(range(base_year, base_year + num_years, 5)), rotation=-45)
			else:
				#print(i,j)

				# hide axes on empty plots
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)


			# global y axes
			if ax.get_subplotspec().is_first_col():
				ax.set_ylabel(a, fontsize = axes_font_size)
				ax.get_yaxis().set_visible(True)

			# global x axes
			if ax.get_subplotspec().is_last_row():
				ax.set_xlabel(b, fontsize = axes_font_size)
				ax.get_xaxis().set_visible(True)

	if name is not None:  
		plt.tight_layout()
		plt.savefig(os.path.join(save_path, name))
		plt.close() 
	else:
		return