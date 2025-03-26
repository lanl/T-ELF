import warnings
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from ..Utility.util import (filter_by, save_fig, create_label_annotate_text, 
                                 create_label_hover_text, format_val, process_label_cols)

# custom jet color scale
# maps bottom 1% of values (0s) to white and other values using the jet color scale
JET_WHITE = [
    (0.0, "white"),     # white color for 0 or nan (bottom 1% of values)
    (0.01, "#000080"),  # start of jet
    (0.1, "#0000FF"),
    (0.3, "#00FFFF"),
    (0.5, "#00FF00"),
    (0.7, "#FFFF00"),
    (0.9, "#FF0000"),
    (1.0, "#800000")   # end of jet
]  

# the factor by which x and y figure sizes are multiplied
# this number can be loosely thought of the as the DPI scaling factor
# it is extracted from a 16:9 aspect ratio producing a 1920x1080 resolution
SCALING_FACTOR = 120

def plot_line(data, x, y, hue, agg_func=None, agg_kwargs=None, base_palette=None, cmap='temps', interactive=True, filters=None, 
              fname=None, n=None, title=None, title_fontsize=14, legend=True, legend_fontsize=12, legend_title_fontsize=14, 
              width=10, xlabel=None, xlabel_fontsize=12, height=6, ylabel=None, ylabel_fontsize=12):
    """
    Plot a line graph from a given dataset with extensive customization options including 
    filtering, aggregation, and styling.

    Parameters
    ----------
    data: pd.DataFrame
        The dataset from which to plot the line graph.
    x: str
        The column name in `data` to be used as the x-axis.
    y: str
        The column name in `data` to be used as the y-axis.
    hue: str
        The column name in `data` to be used for color encoding to represent different categories in the line graph
    agg_func: callable, (optional)
        The aggregation function to be applied to `data` prior to creating the bar plot. Custom Peacock aggregation functions 
        exist under Peacock.Utility.aggregate and Peacock.Utility.aggregate_papers. If not defined, no aggregation will be 
        performed and `x` and `ys` are expected to be in `data`. Default is None.
    agg_kwargs: dict, (optional)
        Additional keyword arguments to be passed to the aggregation function (if defined). Default is None.
    base_palette: dict, (optional)
        A mapping of categories to colors for the plot. If `None`, a palette will be automatically generated.
    cmap: str, (optional)
        The name of the color palette to use if `base_palette` is not provided.
    filters: dict, (optional)
        Criteria to filter `data` before plotting. Keys should be column names, and values should be the values to filter by.
    fname: str, path, None
        The path at which to save the figure. The format is inferred from the extension of fname, if there is one.  If fname does not 
        have a '.png' or '.html' extension, then the file will be saved according to the `interactive` parameter. If not `interactive` 
        a static image is saved as PNG, otherwise a dynamic version is saved as an HTML file. If fname is `None`, the figure is not saved 
        and is instead returned from the function as a plotly figure.
    interactive: bool, (optional)
        If True, output should be an html file. Otherwise a static png image. If `fname` is not defined, this argument will be automatically
        set to True. Default is True.
    legend: bool, (optional)
        If True, render a legend for the plot. Default is True
    legend_fontsize: int, (optional)
        The font size for entries on the legend. Default is 12.
    legend_title_fontsize: int, (optional)
        The font size for entries on the legend. Default is 14.
    n: int, (optional)
        The number of top categories to display based on the sum of `y` values. If `None`, all categories will be displayed.
    title: str, (optional)
        The title of the plot. If `None`, no title is given.
    title_fontsize: int, (optional)
        Font size of the plot title. Default is 24.
    width: int, (optional)
        The x aspect of the figure. `width` * SCALING_FACTOR = number of pixels along x axis. Default is 10.
    xlabel: str, (optional)
        The label for the x-axis of the plot. If not defined then the plot will not have a label for the x-axis. Default is None.
    xlabel_fontsize: str, (optional)
        The font size of the label for the x-axis. Default is 16.
    height: int, (optional)
        The y aspect of the figure. `height` * SCALING_FACTOR = number of pixels along y axis. Default is 6.
    ylabel: str, (optional)
        The label for the y-axis of the plot. If not defined then the plot will not have a label for the y-axis. Default is None.
    ylabel_fontsize: str, (optional)
        The font size of the label for the y-axis. Default is 16.

    Returns
    -------
    plotly.graph_objs._figure.Figure, None
        If `fname` is defined then the generated plot will be saved to disk. Otherwise the function will return the generate Plotly
        figure.
    """
    # rewrite the data instance with a copy
    # plotting function should have read-only access to the original dataframe
    data = data.copy()
    if x == 'year':
        data.year = data.year.astype(float).astype(int)

    if filters:
        selected_data = filter_by(data, filters)
    else:
        selected_data = data.copy()

    if agg_func is not None:
        selected_data = agg_func(selected_data, **agg_kwargs)

    # plot
    return plot_line_helper(base_palette=base_palette, 
                            cmap=cmap,
                            data=selected_data,
                            fname=fname,
                            hue=hue,
                            interactive=interactive,
                            n=n,
                            title=title,
                            title_fontsize=title_fontsize,
                            legend=legend,
                            legend_fontsize=legend_fontsize,
                            legend_title_fontsize=legend_title_fontsize, 
                            x=x, 
                            width=width, 
                            xlabel=xlabel, 
                            xlabel_fontsize=xlabel_fontsize, 
                            y=y, 
                            height=height, 
                            ylabel=ylabel, 
                            ylabel_fontsize=ylabel_fontsize)


def plot_line_helper(base_palette, cmap, data, fname, hue, interactive, n, title, title_fontsize, legend, legend_fontsize, 
                     legend_title_fontsize, x, width, xlabel, xlabel_fontsize, y, height, ylabel, ylabel_fontsize):
    """ 
    Helper function for plot_line(). 
    Do not invoke the helper, use plot_line() instead
    """
    # get the top n samples
    if n is not None:
        top = data.groupby(hue)[y].sum().sort_values(ascending=False).iloc[:n].index.to_list()
        data = data.loc[data[hue].isin(top)].copy()
        
    # color mapping setup
    color_discrete_map = {}
    if base_palette:
        for label in data[hue].unique():
            color_discrete_map[label] = base_palette.get(label, px.colors.qualitative.Plotly[0])

    # create the figure
    fig = px.line(data, 
                  x=x, 
                  y=y, 
                  color=hue, 
                  title=title, 
                  color_discrete_map=color_discrete_map)

    # update axes and layout
    fig.update_layout(
        width=width*SCALING_FACTOR, 
        height=height*SCALING_FACTOR,
        title=dict(x=0.5, text=title, font=dict(size=title_fontsize)),
        showlegend=legend,
        legend=dict(title_text=hue, title_font=dict(size=legend_title_fontsize), font=dict(size=legend_fontsize), itemsizing='constant'),
        xaxis=dict(title=xlabel, title_font=dict(size=xlabel_fontsize)),
        yaxis=dict(title=ylabel, title_font=dict(size=ylabel_fontsize)),
        paper_bgcolor='white', 
        plot_bgcolor='white'
    )

    # save or return the figure
    return save_fig(fig, fname, interactive)

    
def plot_scatter(data, x, y, agg_func=None, agg_kwargs=None, annotate=False, annotate_fontsize=11, base_palette=None, 
                 filters=None, fname=None, hue=None, interactive=True, labels=None, labels_add_scatter_cols=False,
                 legend=True, legend_fontsize=12, legend_title_fontsize=14, log_x=False, log_y=False, log_z=False, 
                 markersize=9, n=None, sort_by=None, title=None, title_fontsize=24, width=10, xlabel=None, 
                 xlabel_fontsize=16, height=6, ylabel=None, ylabel_fontsize=16, z=None, zlabel=None, zlabel_fontsize=16):
    """
    Create a scatter plot from a pandas DataFrame.
    
    This function generates an interactive scatter plot using Plotly with options to color code by groups,
    annotate specific points, and customize visual aspects of the plot. By default the scatter plot will 
    be 2D, however a 3D scatter plot can be created by utilizing the z-axis parameters.

    Parameters
    ----------
    data: pd.DataFrame
        The dataset from which to create the scatter plot
    x: str
        The column name in `data` to be used as the x-axis. Note that if an aggregation function is being applied, the column
        names may change and `x` needs to match the aggregated data.
    y: str
        The column name in `data` to be used as the y-axis. Note that if an aggregation function is being applied, the column
        names may change and `y` needs to match the aggregated data.
    agg_func: callable, (optional)
        The aggregation function to be applied to `data` prior to creating the scatter plot. Custom Peacock aggregation functions 
        exist under Peacock.Utility.aggregate and Peacock.Utility.aggregate_papers. If not defined, no aggregation will be 
        performed and `x` and `y` are expected to be in `data`. Default is None.
    agg_kwargs: dict, (optional)
        Additional keyword arguments to be passed to the aggregation function (if defined). Default is None.
    annotate: bool (optional)
        If True, points on the plot are given annotations. This means that points selected for annotation are given a label that
        is directly drawn on the plot. By default the annotation uses the values from the first column defined in `labels`. If
        `labels` is not provided, the annotation uses the corresponding value for `hue`. If neither are defined and `annotate` is
        True, an error is raised. Default is False.
    annotate_fontsize: int (optional)
        The font size used for text annotations. Default is 11.
    base_palette: dict, (optional)
        A mapping of categories found in `hue` to colors for the plot. Entries in `hue` are keys and the values are color hex codes
        or CSS strings. This palette does not have to be complete: if any values in `hue` do not have a matching color in `base_palette`
        then a new color will be assigned at random. If None, an entire palette will be automatically generated. Default is None.
    filters: dict, (optional)
        Criteria to filter `data` before plotting. Keys should be column names, and values should be the values to filter by.
        Default is None.
    fname: str, path, None
        The path at which to save the figure. The format is inferred from the extension of fname, if there is one.  If fname does not 
        have a '.png' or '.html' extension, then the file will be saved according to the `interactive` parameter. If not `interactive` 
        a static image is saved as PNG, otherwise a dynamic version is saved as an HTML file. If fname is `None`, the figure is not saved 
        and is instead returned from the function as a plotly figure.
    hue: str, (optional)
        The column name in `data` to be used for color encoding to represent different categories in the data. If None, no color 
        categorization will be performed (the plot will have a single trace). Note that if an aggregation function is being applied, 
        the column names may change and `hue` needs to match the aggregated data. Default is None.
    interactive: bool, (optional)
        If True, output should be an html file. Otherwise a static png image. If `fname` is not defined, this argument will be automatically
        set to True. Default is True.
    labels: str, list of str, (optional)
        The columns of `data` that should be used for extra information on the plot. If `labels` is a string it will be converted to a list
        with a single item. The data from the list of columns will be added to the hovertext at each point. If `annotate` is True, then first 
        entry in the `labels` list will be used for annotation. Default is None.
    labels_add_scatter_cols: bool, (optional)
        If True, the values for the numerical attributes of the scatter plot (x, y, and possibly z) will be added to the hover tooltip. 
        Default is False.
    legend: bool, (optional)
        If True, render a legend for the plot. Default is True
    legend_fontsize: int, (optional)
        The font size for entries on the legend. Default is 12.
    legend_title_fontsize: int, (optional)
        The font size for entries on the legend. Default is 14.
    log_x: bool, (optional)
        If True, use logarithmic scale on the x-axis. Default is False.
    log_y: bool, (optional) 
        If True, use logarithmic scale on the y-axis. Default is False.
    markersize: int, (optional)
        The size of the markers used for the scatter plot. Default is 9.
    n: int, (optional)
        How many of the top samples should be annotated (if `annotate` is True) and for how many top samples categories from `hue` should be 
        created. The intention behind this parameter is to limit the details in the legend and annotation of the plot to only the top samples. 
        This should limit how "busy" the plot appears and can prevent annotations from overlapping with each other. Which samples are the 
        "top" samples are determined by `sort_by`. If not defined, all samples in the data are processed. Default is None.
    sort_by: str, (optional)
        The name of the column or columns by which to sort the data to establish the top samples. In practice this variable is probably best
        left untouched as the default behavior of the function is to pick samples that maximize both `x` and `y`. Default is None.
    title: str, (optional)
        The title of the plot. If `None`, no title is given.
    title_fontsize: int, (optional)
        Font size of the plot title. Default is 24.
    width: int, (optional)
        The x aspect of the figure. `width` * SCALING_FACTOR = number of pixels along x axis. Default is 10.
    xlabel: str, (optional)
        The label for the x-axis of the plot. If not defined then the plot will not have a label for the x-axis. Default is None.
    xlabel_fontsize: str, (optional)
        The font size of the label for the x-axis. Default is 16.
    height: int, (optional)
        The y aspect of the figure. `height` * SCALING_FACTOR = number of pixels along y axis. Default is 6.
    ylabel: str, (optional)
        The label for the y-axis of the plot. If not defined then the plot will not have a label for the y-axis. Default is None.
    ylabel_fontsize: str, (optional)
        The font size of the label for the y-axis. Default is 16.
    z: str, (optional)
        The column name in `data` to be used as the z-axis. This parameter along with the rest of z-axis parameters are used if
        creating a 3D scatter plot. If not defined, then a 2D scatter plot is created. Note that if an aggregation function is 
        being applied, the column names may change and `z` needs to match the aggregated data. Default is None.
    zlabel: str, (optional)
        The label for the z-axis of the plot. If not defined then the plot will not have a label for the z-axis. Default is None.
    zlabel_fontsize: str, (optional)
        The font size of the label for the z-axis. Default is 16.

    Returns
    -------
    plotly.graph_objs._figure.Figure, None
        If `fname` is defined then the generated plot will be saved to disk. Otherwise the function will return the generate Plotly
        figure.
    """
    # rewrite the data instance with a copy
    # plotting function should have read-only access to the original dataframe
    data = data.copy()
    
    # filter data if requested
    if filters:
        selected_data = filter_by(data, filters)
    else:
        selected_data = data.copy()

    # if data aggregation function give, perform aggregation
    if agg_func is not None:
        selected_data = agg_func(selected_data, **agg_kwargs)

    # automatically manage `interactive` mode depending on file name/file extension given
    if fname and fname.endswith('.png'):
        if interactive:
            warnings.warn('Requested PNG save format, disabling interactive plot features')
        interactive = False
    elif not fname and not interactive:
        warnings.warn('`fname` not provided. Overriding `interactive` parameter and output will be a Plotly figure')
        interactive = True
    
    # manage plot labels and color
    if hue and not labels:
        labels = process_label_cols(hue)
    elif labels and not hue:
        labels = process_label_cols(labels)
    elif labels and hue:
        labels = process_label_cols(labels)
        labels.append(hue)
    else:
        labels = []
        
    if not hue and not labels and annotate:
        raise ValueError('Cannot annotate plot when `hue` and `labels` are disabled.\n' \
                         'Either disable `annotate` or pass values for `hue` and/or `labels`.')

    # call the plotting helper function to generate 2D scatter plot
    if z is None:
        return plot_scatter_helper(annotate=annotate,
                                   annotate_fontsize=annotate_fontsize,
                                   base_palette=base_palette,
                                   data=selected_data,
                                   filters=filters,
                                   fname=fname,
                                   hue=hue,
                                   interactive=interactive,
                                   labels=labels,
                                   labels_add_scatter_cols=labels_add_scatter_cols,
                                   legend=legend,
                                   legend_fontsize=legend_fontsize,
                                   legend_title_fontsize=legend_title_fontsize,
                                   log_x=log_x,
                                   log_y=log_y,
                                   markersize=markersize,
                                   n=n,
                                   sort_by=sort_by,
                                   title=title,
                                   title_fontsize=title_fontsize,
                                   x=x,
                                   width=width,
                                   xlabel=xlabel,
                                   xlabel_fontsize=xlabel_fontsize,
                                   y=y,
                                   height=height,
                                   ylabel=ylabel,
                                   ylabel_fontsize=ylabel_fontsize)
    else:
        return plot_scatter_3d_helper(annotate=annotate,
                                     annotate_fontsize=annotate_fontsize,
                                     base_palette=base_palette,
                                     data=selected_data,
                                     filters=filters,
                                     fname=fname,
                                     hue=hue,
                                     interactive=interactive,
                                     labels=labels,
                                     labels_add_scatter_cols=labels_add_scatter_cols,
                                     legend=legend,
                                     legend_fontsize=legend_fontsize,
                                     legend_title_fontsize=legend_title_fontsize,
                                     log_x=log_x,
                                     log_y=log_y,
                                     log_z=log_z,
                                     markersize=markersize,
                                     n=n,
                                     sort_by=sort_by,
                                     title=title,
                                     title_fontsize=title_fontsize,
                                     x=x,
                                     width=width,
                                     xlabel=xlabel,
                                     xlabel_fontsize=xlabel_fontsize,
                                     y=y,
                                     height=height,
                                     ylabel=ylabel,
                                     ylabel_fontsize=ylabel_fontsize,
                                     z=z,
                                     zlabel=zlabel,
                                     zlabel_fontsize=zlabel_fontsize)


def plot_scatter_helper(annotate, annotate_fontsize, base_palette, data, filters, fname, hue, interactive, labels, 
                        labels_add_scatter_cols, legend, legend_fontsize, legend_title_fontsize, log_x, log_y, markersize, 
                        n, sort_by, title, title_fontsize, x, width, xlabel, xlabel_fontsize, y, height, ylabel, 
                        ylabel_fontsize):
    """ 
    Helper function for plot_scatter(). 
    Do not invoke the helper, use plot_scatter() instead
    """
    fig = go.Figure()
    if n:
        if sort_by is None:
            sort_by = [x, y]  # default to sorting by both x and y if no sort_by provided
        top_data = data.nlargest(n, columns=sort_by)
        other_data = data.drop(top_data.index)
    else:
        top_data = data
        other_data = pd.DataFrame(columns=data.columns)  # empty dataframe

    # color mapping setup
    color_discrete_map = {}
    if base_palette:
        for label in top_data[hue].unique():
            color_discrete_map[label] = base_palette.get(label, px.colors.qualitative.Plotly[0])
        color_discrete_map['Other'] = 'grey'  # generic color for non-top labels

    # add x and y to labels columns if requested
    if labels_add_scatter_cols:
        labels += [x,y]
        
    # hue column was set
    # plot traces for each of the top authors (all if n is None)
    # samples outside top n will be given 'Other' label
    for label, df in [('Top', top_data), ('Other', other_data)]:
        if label == 'Top':
            hue_labels = df[hue].unique() if hue else ['Trace']  # get hue labels if hue provided, otherwise dummy value
            for hue_label in hue_labels:
                if hue:
                    filtered_data = df[df[hue] == hue_label]  # subset of data that contains hue label
                else:
                    filtered_data = df
                annotate_text = filtered_data.apply(lambda row: create_label_annotate_text(row, labels), axis=1) if labels else None
                hover_text = filtered_data.apply(lambda row: create_label_hover_text(row, labels), axis=1) if labels else None
                fig.add_trace(go.Scatter(
                    x=filtered_data[x],
                    y=filtered_data[y],
                    mode='markers+text' if annotate else 'markers',
                    marker=dict(size=markersize, color=color_discrete_map.get(hue_label) if hue else 'grey'),
                    name=hue_label,
                    text=annotate_text if annotate else None,
                    textfont=dict(size=annotate_fontsize, color='black'),
                    textposition='top center',
                    hoverinfo='text',
                    hovertext=hover_text,
                    showlegend=bool(hue)
                ))
        else:
            hover_text = df.apply(lambda row: create_label_hover_text(row, labels), axis=1) if labels else None
            fig.add_trace(go.Scatter(
                x=df[x],
                y=df[y],
                mode='markers',
                marker=dict(size=markersize, color='grey'),
                name='Other',
                text=hover_text,
                hoverinfo='text',
                showlegend=bool(hue)
            ))
    
    # if hue not provided, not traces exist, no need to show legend regardless of `legend` parameter
    if not hue:
        fig.update_layout(showlegend=False)
    
    # update axes and layout
    fig.update_layout(
        width=width*SCALING_FACTOR, 
        height=height*SCALING_FACTOR,
        title=dict(x=0.5, text=title, font=dict(size=title_fontsize)),
        showlegend=legend,
        legend=dict(title_text=hue, title_font=dict(size=legend_title_fontsize), font=dict(size=legend_fontsize), itemsizing='constant'),
        xaxis=dict(title=xlabel, title_font=dict(size=xlabel_fontsize)),
        yaxis=dict(title=ylabel, title_font=dict(size=ylabel_fontsize)),
        paper_bgcolor='white', 
        plot_bgcolor='white'
    )

    # add border to plot
    fig.update_xaxes(showline=True,
             linewidth=1,
             linecolor='black',
             mirror=True)

    fig.update_yaxes(showline=True,
             linewidth=1,
             linecolor='black',
             mirror=True)
    
    # change scale to log if requested
    if log_x:
        fig.update_xaxes(type='log')
    if log_y:
        fig.update_yaxes(type='log')

    # save or return the figure
    return save_fig(fig, fname, interactive)

    
def plot_scatter_3d_helper(annotate, annotate_fontsize, base_palette, data, filters, fname, hue, interactive, labels, 
                           labels_add_scatter_cols, legend, legend_fontsize, legend_title_fontsize, log_x, log_y, log_z, 
                           markersize, n, sort_by, title, title_fontsize, x, width, xlabel, xlabel_fontsize, y, height, 
                           ylabel, ylabel_fontsize, z, zlabel, zlabel_fontsize):
    """ 
    Helper function for plot_scatter(). 
    Do not invoke the helper directly; use plot_scatter() instead.
    """
    fig = go.Figure()
    if n:
        if sort_by is None:
            sort_by = [x, y, z]  # default to sorting by x, y, and z if no sort_by provided
        top_data = data.nlargest(n, columns=sort_by)
        other_data = data.drop(top_data.index)
    else:
        top_data = data
        other_data = pd.DataFrame(columns=data.columns)  # empty dataframe

    # color mapping setup
    color_discrete_map = {}
    if base_palette:
        for label in top_data[hue].unique():
            color_discrete_map[label] = base_palette.get(label, px.colors.qualitative.Plotly[0])
        color_discrete_map['Other'] = 'grey'  # generic color for non-top labels

    # add x, y, and z to labels columns if requested
    if labels_add_scatter_cols:
        labels += [x,y,z]
        
    # creating traces for the plot
    for label, df in [('Top', top_data), ('Others', other_data)]:
        if label == 'Top':
            hue_labels = df[hue].unique() if hue else ['Trace']  # get hue labels if hue provided, otherwise dummy value
            for hue_label in hue_labels:
                if hue:
                    filtered_data = df[df[hue] == hue_label]  # subset of data that contains hue label
                else:
                    filtered_data = df
                annotate_text = filtered_data.apply(lambda row: create_label_annotate_text(row, labels), axis=1) if labels else None
                hover_text = filtered_data.apply(lambda row: create_label_hover_text(row, labels), axis=1) if labels else None
                fig.add_trace(go.Scatter3d(
                    x=filtered_data[x],
                    y=filtered_data[y],
                    z=filtered_data[z],
                    mode='markers+text' if annotate else 'markers',
                    marker=dict(size=markersize, color=color_discrete_map.get(hue_label) if hue else 'grey'),
                    name=hue_label,
                    text=annotate_text if annotate else None,
                    textfont=dict(size=annotate_fontsize, color='black'),
                    hoverinfo='text',
                    hovertext=hover_text,
                    showlegend=bool(hue)
                ))
        else:
            hover_text = df.apply(lambda row: create_label_hover_text(row, labels), axis=1) if labels else None
            fig.add_trace(go.Scatter3d(
                x=df[x],
                y=df[y],
                z=df[z],
                mode='markers',
                marker=dict(size=markersize, color='grey'),
                name='Other',
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=bool(hue)
            ))

    # if hue not provided, not traces exist, no need to show legend regardless of `legend` parameter
    if not hue:
        fig.update_layout(showlegend=False)
            
    # update axes and layout for 3D
    fig.update_layout(
        width=width*SCALING_FACTOR, 
        height=height*SCALING_FACTOR,
        title=dict(x=0.5, text=title, font=dict(size=title_fontsize)),
        showlegend=legend,
        legend=dict(title_text=hue, title_font=dict(size=legend_title_fontsize), font=dict(size=legend_fontsize), itemsizing='constant'),
        scene=dict(
            xaxis=dict(title=xlabel, title_font=dict(size=xlabel_fontsize)),
            yaxis=dict(title=ylabel, title_font=dict(size=ylabel_fontsize)),
            zaxis=dict(title=zlabel, title_font=dict(size=zlabel_fontsize))
        ),
        paper_bgcolor='white', 
        plot_bgcolor='white'
    )

    # log scale settings for axes
    if log_x:
        fig.update_layout(scene=dict(xaxis=dict(type='log')))
    if log_y:
        fig.update_layout(scene=dict(yaxis=dict(type='log')))
    if log_z:
        fig.update_layout(scene=dict(zaxis=dict(type='log')))

    # save or return the figure
    return save_fig(fig, fname, interactive)


def plot_heatmap(data, annotate=False, annotate_fontsize=11, cmap='portland', colorbar=True, colorbar_title=None, colorbar_title_fontsize=None, 
                 colorbar_ticks_fontsize=None, fname=None, highlight=[], interactive=True, interpolation='bilinear', labels=None, 
                 title=None, title_fontsize=24, width=10, xlabel=None, xlabel_fontsize=16, height=6, ylabel=None, ylabel_fontsize=16):
    """
    Create a heatmap of the data.
    
    This function generates an interactive heatmap plot using Plotly with options for adding metadata on hover and
    higlights of rows

    Parameters
    ----------
    data: pd.DataFrame, np.ndarray
        The dataset from which to create the heatmap plot. If DataFrame, it is expected to be a pivot table with indices going on
        y axis and columns going on x axis. If a numpy array, it should be 2D and attributes will be numbered.
    annotate: bool (optional)
        If True, heatmap cells are given annotations. The value of the cell is written in text over top of the cell. Default is
        False.
    cmap: str, (optional)
        The continuous color scale map to use for the heatmap. The values of `cmap` should be strings that correspond to color scales
        defined in plotly.express.colors.named_colorscales(). The one exception to this is the string 'jet_white' which is a custom
        continuous color scale defind in Peacock that maps the bottom 1% of values to white. Default is 'portland'.
    colorbar: bool, (optional)
        If True, show the colorbar with the heatmap. Default is True.
    colorbar_title: str, (optional)
        Set the title of the colorbar. If None the word 'Colorbar' is used. To remove title but keep the colorbar, pass an empty string.
        Default is None.
    colorbar_title_fontsize: int, (optional)
        The size of the colobar title. If None, use Plotly automatic sizing. Default is None.
    colorbar_ticks_fontsize
        The size of the colorbar ticks. If None, use Plotly automatic sizing. Default is None.
    fname: str, path, None
        The path at which to save the figure. The format is inferred from the extension of fname, if there is one.  If fname does not 
        have a '.png' or '.html' extension, then the file will be saved according to the `interactive` parameter. If not `interactive` 
        a static image is saved as PNG, otherwise a dynamic version is saved as an HTML file. If fname is `None`, the figure is not saved 
        and is instead returned from the function as a plotly figure.
    interactive: bool, (optional)
        If True, output should be an html file. Otherwise a static png image. If `fname` is not defined, this argument will be automatically
        set to True. Default is True.
    highlight: list, (optional)
        A list of items that should be highlighted in the heatmap. A 'highlight' is a red line through the row featuring the item. If `data`
        is a pandas DataFrame then the values in `highlight` should correspond to the index values of the DataFrame. If `data` is an numpy
        array then `higlight` values should be the numerical index values. Default is an empty list (no highlighting). 
    labels: dict, (optional)
        A supplementary map that can add to the heatmap hover info. This map takes the format of a dict of dicts where the top level dictionary
        contains indices as keys and subdicts of attributes as values. For example, a pandas DataFrame that is indexed by the letters of the 
        alphabet may have a map of the following structure:
        >>>labels = {
                        "a": {"title": 'foo', "description": 'foo description'},
                        "b": {"title": 'bar', "description": 'bar description'},
                        "c": {"title": 'buzz', "description": 'buzz description'},
                    }
        This would then generate supplemetary information for the rows 'a', 'b', and 'c', providing title and descrition information for each in
        the hoper tooltip. If no labels are given, no extra attributes are generated and only `data` is used for the heatmap generation. 
        Default is None.
    title: str, (optional)
        The title of the plot. If `None`, no title is given.
    title_fontsize: int, (optional)
        Font size of the plot title. Default is 24.
    width: int, (optional)
        The x aspect of the figure. `width` * SCALING_FACTOR = number of pixels along x axis. Default is 10.
    xlabel: str, (optional)
        The label for the x-axis of the plot. If not defined then the plot will not have a label for the x-axis. Default is None.
    xlabel_fontsize: str, (optional)
        The font size of the label for the x-axis. Default is 16.
    height: int, (optional)
        The y aspect of the figure. `height` * SCALING_FACTOR = number of pixels along y axis. Default is 6.
    ylabel: str, (optional)
        The label for the y-axis of the plot. If not defined then the plot will not have a label for the y-axis. Default is None.
    ylabel_fontsize: str, (optional)
        The font size of the label for the y-axis. Default is 16.

    Returns
    -------
    plotly.graph_objs._figure.Figure, None
        If `fname` is defined then the generated plot will be saved to disk. Otherwise the function will return the generate Plotly
        figure.
    """
    # rewrite the data instance with a copy
    # plotting function should have read-only access to the original dataframe
    data = data.copy()

    # automatically manage `interactive` mode depending on file name/file extension given
    if fname and fname.endswith('.png'):
        if interactive:
            warnings.warn('Requested PNG save format, disabling interactive plot features')
        interactive = False
    elif not fname and not interactive:
        warnings.warn('`fname` not provided. Overriding `interactive` parameter and output will be a Plotly figure')
        interactive = True
    
    # select the custom cmap if requested
    if cmap == 'jet_white':
        cmap = JET_WHITE
        
    return plot_heatmap_helper(annotate=annotate,
                               fname=fname, 
                               cmap=cmap, 
                               colorbar=colorbar,
                               colorbar_title=colorbar_title,
                               colorbar_title_fontsize=colorbar_title_fontsize,
                               colorbar_ticks_fontsize=colorbar_ticks_fontsize,
                               data=data,
                               highlight=highlight,
                               interactive=interactive, 
                               interpolation=interpolation, 
                               labels=labels,
                               title=title, 
                               title_fontsize=title_fontsize, 
                               width=width, 
                               xlabel=xlabel, 
                               xlabel_fontsize=xlabel_fontsize,
                               height=height, 
                               ylabel=ylabel, 
                               ylabel_fontsize=ylabel_fontsize)
    
    
def plot_heatmap_helper(annotate, fname, cmap, colorbar, colorbar_title, colorbar_title_fontsize, colorbar_ticks_fontsize, 
                        data, highlight, interactive, interpolation, labels, title, title_fontsize, width, xlabel, 
                        xlabel_fontsize, height, ylabel, ylabel_fontsize):
    """ 
    Helper function for plot_heatmap(). 
    Do not invoke the helper directly; use plot_heatmap() instead.
    """
    # check if data is DataFrame
    if isinstance(data, pd.DataFrame):
        x_labels = data.columns
        y_labels = data.index
    elif isinstance(data, np.ndarray):
        x_labels = list(range(data.shape[1]))
        y_labels = list(range(data.shape[0]))
        data = pd.DataFrame(data, index=y_labels, columns=x_labels)
    else:
        raise TypeError("`data` must be a pd.DataFrame or an np.ndarray.")
    
    # prepare hover text with cell values and additional information
    hover_texts = []
    for row_idx, row in data.iterrows():
        hover_row = []
        for col_idx, value in row.items():
            base_text = f'{value}<br>({row_idx}, {col_idx})'
            if labels and labels.get(row_idx, {}):
                
                # create the hovertext for each itemm\
                extra_info = '<br>' + '<br>'.join([f'<b>{k}</b>: {v}' for k, v in labels[row_idx].items()])
                hover_text = base_text + extra_info
            else:
                hover_text = base_text
            hover_row.append(hover_text)
        hover_texts.append(hover_row)
    
    # colorbar settings
    colorbar_dict = {
        'title': colorbar_title,
        'titlefont': {'size': colorbar_title_fontsize} if colorbar_title_fontsize else None,
        'tickfont': {'size': colorbar_ticks_fontsize} if colorbar_ticks_fontsize else None
    } if colorbar else None
        
    # create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=x_labels,
        y=y_labels,
        colorscale=cmap,
        text=hover_texts if annotate else None,
        hoverinfo='text',
        hovertext=hover_texts,
        colorbar=colorbar_dict
    ))
    
    # highlight specific rows if provided
    if highlight:
        for highlight_idx in highlight:
            if highlight_idx in y_labels:
                real_index = list(y_labels).index(highlight_idx)  # find actual index for highlighting
                fig.add_shape(type='line', 
                              x0=min(x_labels) - 0.5, x1=max(x_labels) - 0.5, 
                              y0=real_index, y1=real_index, 
                              line=dict(color='red', width=2),
                              opacity=0.2)
    
    # update axes and layout
    fig.update_layout(
        width=width*SCALING_FACTOR, 
        height=height*SCALING_FACTOR,
        title=dict(x=0.5, text=title, font=dict(size=title_fontsize)),
        xaxis=dict(
            title=xlabel, 
            title_font=dict(size=xlabel_fontsize),
            tickmode='array',
        ),
        yaxis=dict(
            title=ylabel, 
            title_font=dict(size=ylabel_fontsize),
            tickmode='array',
        ),
        paper_bgcolor='white', 
        plot_bgcolor='white'
    )
    
    # add border to plot
    fig.update_xaxes(showline=True,
             linewidth=1,
             linecolor='black',
             mirror=True)

    fig.update_yaxes(showline=True,
             linewidth=1,
             linecolor='black',
             mirror=True)
    
    # save or return the figure
    return save_fig(fig, fname, interactive)


def plot_bar(data, x, ys, agg_func_name=None, agg_kwargs=None, annotate=False, annotate_color='black', annotate_fontsize=11, cmap='rdylbu', 
             color='value', colorbar=True, filters=None, fname=None, interactive=True, n=None, labels=None, sort_by=None, ys_fontsize=12, 
             title=None, title_fontsize=24, width=10, xlabel=None, ylabel=None, xlabel_fontsize=16, height=6, ylabel_fontsize=16):
    """
    Create a bar plot of the data.
    
    This function generates an interactive barplot plot using Plotly with options for adding metadata on hover and
    annotatoins directly on the plot. 

    Parameters
    ----------
    data: pd.DataFrame, np.ndarray
        The dataset from which to create the bar plot.
    x: str
        The column name in `data` to be used as the x-axis. These are the samples/categories.
    ys: str, list
        The column name(s) in `data` to be used as the y-axis. These are the value(s) to be show on the bar plot. If this is a list
        of values then the first value will be used as the value for the bar plot and the other columns will be displayed in the
        hoverdata of the plot and optionally the annotation (if `annotate` is True)
    agg_func: callable, (optional)
        The aggregation function to be applied to `data` prior to creating the bar plot. Custom Peacock aggregation functions 
        exist under Peacock.Utility.aggregate and Peacock.Utility.aggregate_papers. If not defined, no aggregation will be 
        performed and `x` and `ys` are expected to be in `data`. Default is None.
    agg_kwargs: dict, (optional)
        Additional keyword arguments to be passed to the aggregation function (if defined). Default is None.
    annotate: bool (optional)
        If True, the bars on the plot are given annotations. This means that if more than one column is provided in `ys`, the 
        additional column data is not provided separate bars but rather annotate directly on the existing bar for the 
        corresponding sample. This data would exist in the hover information of the bar plot but setting `annotate` to True
        can preserve this information if exporting the plot in a static format. Default is False.
    annotate_color: str, (optional)
        The color of the annotation text. Expects a hex code or a CSS color string. Default is 'black.
    annotate_fontsize: int (optional)
        The font size used for text annotations. Default is 11.
    cmap: str, (optional)
        The continuous color scale map to use for the heatmap. The values of `cmap` should be strings that correspond to color scales
        defined in plotly.express.colors.named_colorscales(). The one exception to this is the string 'jet_white' which is a custom
        continuous color scale defind in Peacock that maps the bottom 1% of values to white. Note that if `color` is set to 'value' 
        (the default), then the color map is used. Otherwise, the colormap is overriden. Default is 'rdylbu'. 
    color: str, (optional)
        The property that should be used to establish the color of the plot. This parameter takes three values ['value', 'sample', None]. 
        If set to 'value', the property being plotted in the bar plot (the firt entry in `ys`) will be used to establish a continuous
        color scale. If set to 'sample', each sample for which a bar is being generated is given an individual color. If set to None, 
        all the bars are given the same color. Default is 'value'.
    colorbar: bool, (optional)
        If True, show the colorbar for the bar plot when `color` == 'value'. If instead using `color` == 'sample' then this parameter 
        will control if the legend is generated. Default is True.
    filters: dict, (optional)
        Criteria to filter `data` before plotting. Keys should be column names, and values should be the values to filter by.
        Default is None.
    fname: str, path, None
        The path at which to save the figure. The format is inferred from the extension of fname, if there is one.  If fname does not 
        have a '.png' or '.html' extension, then the file will be saved according to the `interactive` parameter. If not `interactive` 
        a static image is saved as PNG, otherwise a dynamic version is saved as an HTML file. If fname is `None`, the figure is not saved 
        and is instead returned from the function as a plotly figure.
    interactive: bool, (optional)
        If True, output should be an html file. Otherwise a static png image. If `fname` is not defined, this argument will be automatically
        set to True. Default is True.
    labels: str, list of str, (optional)
        The columns of `data` that should be used for extra information on the plot. If `labels` is a string it will be converted to a list
        with a single item. The data from the list of columns will be added to the hovertext at each point. If `annotate` is True, then first 
        entry in the `labels` list will be used for annotation. Default is None.
    n: int, (optional)
        How many of the top samples should be selected if aggregation is being performed. Which samples are the 
        "top" samples are determined by `sort_by`. If not defined, all samples in the data are processed. Default is None.
    sort_by: str, (optional)
        The name of the column or columns by which to sort the data to establish the top samples. In practice this variable is probably best
        left untouched as the default behavior of the function is to pick samples that maximize both `x` and `y`. Default is None.
    title: str, (optional)
        The title of the plot. If `None`, no title is given.
    title_fontsize: int, (optional)
        Font size of the plot title. Default is 24.
    width: int, (optional)
        The x aspect of the figure. `width` * SCALING_FACTOR = number of pixels along x axis. Default is 10.
    xlabel: str, (optional)
        The label for the x-axis of the plot. If not defined then the plot will not have a label for the x-axis. Default is None.
    xlabel_fontsize: str, (optional)
        The font size of the label for the x-axis. Default is 16.
    height: int, (optional)
        The y aspect of the figure. `height` * SCALING_FACTOR = number of pixels along y axis. Default is 6.
    ylabel: str, (optional)
        The label for the y-axis of the plot. If not defined then the plot will not have a label for the y-axis. Default is None.
    ylabel_fontsize: str, (optional)
        The font size of the label for the y-axis. Default is 16.    
    """
    # rewrite the data instance with a copy
    # plotting function should have read-only access to the original dataframe
    data = data.copy()
        
    if filters:
        selected_data = filter_by(data, filters)
    else:
        selected_data = data.copy()
        
    # validate input for labels
    labels = process_label_cols(labels)
    
    # validate input for ys
    ys = process_label_cols(ys)
    
    # validate color map anchor
    if color == 'sample':
        color = x
    elif color == 'value':
        color = ys[0]
    else:
        raise ValueError("Unknown value for `color`. Options are ['sample', 'value']")
    
    # sort values by the specified order
    if sort_by is not None:
        selected_data = selected_data.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

    return plot_bar_helper(annotate=annotate,
                           annotate_fontsize=annotate_fontsize,
                           annotate_color=annotate_color,
                           cmap=cmap,
                           color=color,
                           colorbar=colorbar,
                           data=selected_data, 
                           fname=fname, 
                           interactive=interactive,
                           labels=labels,
                           title=title, 
                           title_fontsize=title_fontsize, 
                           x=x,
                           width=width,
                           xlabel=xlabel, 
                           xlabel_fontsize=xlabel_fontsize,
                           ys=ys,
                           height=height,
                           ylabel=ylabel, 
                           ylabel_fontsize=ylabel_fontsize)

    
def plot_bar_helper(annotate, annotate_fontsize, annotate_color, cmap, color, colorbar, data, fname, interactive, labels, 
                    title, title_fontsize, x, width, xlabel, xlabel_fontsize, ys, ylabel, height, ylabel_fontsize):
    """ 
    Helper function for plot_bar(). 
    Do not invoke the helper directly; use plot_bar() instead.
    """
    data = data.copy()
    primary_y = ys.pop(0)  # remove first element of ys list
    hover_data = {primary_y: True}

    # approach needed to ensure that annotations for numeric x values are properly handled
    # plotly won't treat numerical data as categorical no matter if data is string
    # approach is to add invisible character (U+2800 BRAILLE BLOCK) to force plotly to treat data as categorical
    x_is_numeric = np.array([(np.isreal(val) or val.isnumeric()) for val in data[x].values.flatten()]).all()
    if annotate and x_is_numeric:
        
        data[x] = data[x].astype(str)           
        data[x] = data[x] + '⠀'
    
    # add hover data
    if labels:
        hover_data.update({k: True for k in labels})
    if ys:
        hover_data.update({k: True for k in ys})
    
    # sort data before plotting
    data = data.sort_values(by=primary_y, ascending=False)
    
    # create bar plot
    fig = px.bar(data, 
                 x=x, 
                 y=primary_y,
                 title=title, 
                 color=color,
                 color_continuous_scale=cmap,
                 hover_data=hover_data)

    # if annotations are being added, add them in this conditional
    title_centering_factor = 0.5
    if annotate:
        
        # variables to adjust the position of annotations
        offset_factor = max(data[primary_y]) / 20
        attribute_name_y_pos = {}

        # add annotations for additional ys values on the bars
        for index, row in data.iterrows():
            for idx, y in enumerate(ys):
                
                y_pos = (idx + 1) * offset_factor # calculate annotation position
                fig.add_annotation(x=row[x], y=y_pos, text=f'{format_val(row[y])}',
                                   showarrow=False, font=dict(size=annotate_fontsize, color=annotate_color),
                                   xanchor='center', yanchor='bottom')

                # add one annotation for attribute name outside the bars
                if y not in attribute_name_y_pos:
                    attribute_name_y_pos[y] = y_pos  


        # calculate char width normalization factor based on plot width
        max_length_y = max(ys, key=len)  # find the longest y attribute name for width estimation
        total_width_px = width * SCALING_FACTOR  # total width in pixels
        normalized_char_width = total_width_px / 100  # normalize based on empirical factor (magic number that works :))
        estimated_text_width = len(max_length_y) * normalized_char_width * (annotate_fontsize / 12)  # adjust estimation based on annotation font size
        fig.update_layout(margin=dict(l=estimated_text_width))  # adjust left margin by scaling factor
    
        # add names outside of the figure
        for y, pos in attribute_name_y_pos.items():
            fig.add_annotation(text=y, xref="paper", yref="y",
                               x=-0.05, y=pos, showarrow=False,
                               font=dict(size=annotate_fontsize+1, color=annotate_color), xanchor='right', yanchor='middle')
        
        # adjust the location of the title 
        # this adjustment centers the title for most cases of annotation font size and figure size
        # the adjustment is not perfect, just a heuristic, and could be adjusted to perfectly center the title
        title_centering_factor -= 0.03
        title_centering_factor += (estimated_text_width / (total_width_px + estimated_text_width)) / 2
        
    # update axes and layout
    fig.update_layout(
        width=width*SCALING_FACTOR, 
        height=height*SCALING_FACTOR,
        title=dict(x=title_centering_factor, text=title, font=dict(size=title_fontsize)),
        xaxis=dict(title=xlabel, title_font=dict(size=xlabel_fontsize)),
        yaxis=dict(title=ylabel, title_font=dict(size=ylabel_fontsize)),
        showlegend=colorbar,
        coloraxis_showscale=colorbar,
        paper_bgcolor='white', 
        plot_bgcolor='white'
    )
    
    # add border to plot
    fig.update_xaxes(showline=True,
                     linewidth=1,
                     linecolor='black',
                     mirror=True)

    fig.update_yaxes(showline=True,
                     linewidth=1,
                     linecolor='black',
                     mirror=True)


    # save or return the figure
    return save_fig(fig, fname, interactive)


def plot_pie(data, *, autotext_fontsize=12, cmap='tab20', colors=None, default_color='auto', explode_other=False, fname=None, 
             height=8, other_thresh=0.035, pie_font=None, startangle=140, title=None, title_fontsize=24, wc_font=None, width=8, 
             words=None):
    """
    Plot a pie chart with an optional word cloud in the center.

    Parameters:
    -----------
    data: dict
        Data dictionary where keys are categories and values are the corresponding magnitudes.
    autotext_fontsize: int
        Font size of the percentage texts on the pie chart. Default is 12.
    cmap: str
        Matplotlib colormap name used when `default_color` is set to 'auto'.
    colors: dict, (optional)
        Dictionary mapping categories to colors. If no map is provided, then `default_color` is used for assigning color to
        the pie chart. Default is None.
    default_color: str, (optional)
        Default color for pie segments that aren't explicitly assigned a color. Can be 'auto' for automatic color assignment.
    explode_other: bool, (optional)
        If True, explode the 'Other' category in the pie chart.
    fname: str, path, None, (optional)
        The path at which to save the figure. The format is inferred from the extension of fname, if there is one. If fname has no 
        extension, then the file is saved with matplotlib.rcParams["savefig.format"] (default: 'png') and the appropriate extension is 
        appended to fname. If fname is `None`, the figure is not saved and is instead returned to be viewed in a jupyter notebook.
        Default is None.
    height: int, (optional)
        Height of the figure in inches. Default is 8.
    other_thresh: float, (optional)
        Threshold percentage to group smaller segments into 'Other'. If a category value / sum of all category values is less than 
        `other_thresh` then said category will be moved into 'Other'. Set this threshold to 0 to avoid such binning. Default is 0.035.
    pie_font: str, (optional)
        Font family specifically for the pie chart text. This setting will adjust the font of all parts of the pie chart except the 
        inner wordcloud. If None, the matplotlib.pyplot default font will be used. Default is None. 
    startangle: int, (optiona)
        Start angle for the pie slices. Default is 140. 
    title: str, (optinal)
        The title of the pie chart. Default is None.
    title_fontsize: int, (optinal)
        The fontsize of the title of the pie chart. Default is 24.
    wc_font: str, (optional)
        Font for the wordcloud figure embedded in the center of the pie chart. If not defined, the wordcloud library default is used.
        Default is None.
    width: int, (optional)
        Width of the figure in inches. Default is 8.
    words: dict (optional)
        Word frequencies for generating a word cloud. This should be a dictionary of word keys and frequency values. If not defined,
        no wordcloud will be generated. Default is None.

    Returns:
    --------
    None: 
        Displays or saves the output pie chart
    """
    if pie_font is not None:
        plt.rcParams['font.family'] = pie_font

    # calculate the threshold for the bottom percentage
    total_value = sum(data.values())
    
    main_data = {}
    other_total = 0

    for key, value in data.items():
        representation = value / total_value
        if representation < other_thresh:
            other_total += value
        else:
            main_data[key] = value

    # add 'Other' category if needed
    if other_total > 0:
        main_data['Other'] = other_total

    # setup colors
    if colors is None:
        colors = {}

    # establish different colors for each category that is not represented in `colors`
    if default_color == 'auto':
        unique_colors = plt.get_cmap(cmap)  # Get colormap from matplotlib
        color_keys = list(main_data.keys())
        pie_colors = [unique_colors(i % unique_colors.N) for i in range(len(color_keys))]  # Use modulo to cycle through colors
    else:  # use the default color for all missing categories
        pie_colors = [colors.get(key, default_color) for key in main_data.keys()]

    # set up explode settings if specified
    explode = [0.1 if label == 'Other' and explode_other else 0 for label in main_data.keys()]

    # create pie chart
    fig, ax = plt.subplots(figsize=(width, height))
    wedges, texts, autotexts = ax.pie(main_data.values(), labels=main_data.keys(), colors=pie_colors,
                                      autopct='%1.1f%%', pctdistance=0.85, startangle=startangle,
                                      wedgeprops=dict(width=0.3), explode=explode)

    # optionally create and embed wordcloud
    if words:
        w, h = width * 100, height * 100
        x, y = np.ogrid[:w, :h]
        mask = (x - w // 2) ** 2 + (y - h // 2) ** 2 > (int(w * 0.45)) ** 2
        mask = 255 * mask.astype(int)
        wc = WordCloud(mask=mask, width=w, height=h, max_words=100, background_color="white").generate_from_frequencies(words)
        ax.imshow(wc, aspect='auto', extent=(-0.65, 0.65, -0.65, 0.65))

    # adjust plot
    ax.axis("off") 
    plt.title(title, fontsize=title_fontsize, fontweight='bold')
    plt.setp(wedges, linewidth=(width * height) / 128, edgecolor='white')
    plt.setp(autotexts, size=autotext_fontsize, weight="bold", color='white')
    
    # save or display the plot
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        