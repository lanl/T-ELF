import warnings
import numpy as np
import pandas as pd

def nunique(data, target_column, group_columns=None):
    """
    Group a dataframe by the column names listed in group_columns (if provided) and then find the number
    of unique instances of values in the target_column for each group. If group_columns is not provided,
    it will simply return the number of unique values in the target_column.


    Parameters
    ----------
    data: pd.DataFrame
        Data from which to aggregate
    target_column: str
        The column for which to count the unique values per group
    group_columns: list, (optional)
        List of column names in data by which to group the results. These must be 
        valid column names in data

    Returns
    -------
    aggregated_data: pd.DataFrame
        The aggregated results from data
    
    Example
    -------
    >>> paper_count_by_year_per_cluster = aggregate_nunique(data, 'id', ['year', 'cluster'])

    """
    if group_columns:
        return data.groupby(group_columns)[target_column].nunique().reset_index()
    else:
        return data[target_column].nunique()


def sum(data, group_columns, top_n=None, sort_by=None, round_floats=3, preserve=['author', 'author_id']):
    """
    Group a dataframe by the column names listed in group_columns. Then for every numeric variable in 
    data, compute the sum. Return the resulting DataFrame.

    Parameters
    ----------
    data: pd.DataFrame
        Data from which to aggregate
    group_columns: list
        List of column names in data by which to group the results. These must be 
        valid column names in data
    top_n: int
        The top n originators (by paper count) which to include in the output. 
        Default is None.
    sort_by: str
        Column name by which to sort to get the top values if top_n is not None
        Default is None    
    round_floats: bool
        How many values to which to round the results of the aggregation. Perform no
        rounding by setting this value to None. Default is 3
    preserve: list
        List of column names that should be preserved during aggregation. The first value
        seen for these columns in each aggregation group is used. An example use case 
        may be seen when group_columns contains 'author_id'. If 'author' is not in 
        preserve, only 'author_id' will determine the identity of the author in the 
        aggregated results. Since each 'author' and 'author_id' pair should be unique,
        using the first seen author name for an arbitrarty author id will accurately 
        preserve this information. Note that columns in preserve should be distinct 
        from columns in group_columns. Default = ['author', 'author_id'].
        
    Returns
    -------
    aggregated_data: pd.DataFrame
        The aggregated results from data
    """
    return _aggregate_numpy(data=data,
                            func='sum',
                            group_columns=group_columns,
                            top_n=top_n,
                            sort_by=sort_by,
                            round_floats=round_floats,
                            preserve=preserve,
                           )


def mean(data, group_columns, top_n=None, sort_by=None, round_floats=3, preserve=['author', 'author_id']):
    """
    Group a dataframe by the column names listed in group_columns. Then for every numeric variable in 
    data, compute the mean. Return the resulting DataFrame.

    Parameters
    ----------
    data: pd.DataFrame
        Data from which to aggregate
    group_columns: list
        List of column names in data by which to group the results. These must be 
        valid column names in data
    top_n: int
        The top n originators (by paper count) which to include in the output. 
        Default is None.
    sort_by: str
        Column name by which to sort to get the top values if top_n is not None
        Default is None    
    round_floats: bool
        How many values to which to round the results of the aggregation. Perform no
        rounding by setting this value to None. Default is 3
    preserve: list
        List of column names that should be preserved during aggregation. The first value
        seen for these columns in each aggregation group is used. An example use case 
        may be seen when group_columns contains 'author_id'. If 'author' is not in 
        preserve, only 'author_id' will determine the identity of the author in the 
        aggregated results. Since each 'author' and 'author_id' pair should be unique,
        using the first seen author name for an arbitrarty author id will accurately 
        preserve this information. Note that columns in preserve should be distinct 
        from columns in group_columns. Default = ['author', 'author_id'].
        
    Returns
    -------
    aggregated_data: pd.DataFrame
        The aggregated results from data
    
    Example
    -------
    >>> aggregated_data = aggregate_mean(data=my_data, 
                                         group_columns=['author_id'], 
                                         top_n=10, 
                                         sort_by='indegree', 
                                         round_floats=3, 
                                         preserve=['author', 'affiliation', 'country'])
    """
    return _aggregate_numpy(data=data,
                            func='mean',
                            group_columns=group_columns,
                            top_n=top_n,
                            sort_by=sort_by,
                            round_floats=round_floats,
                            preserve=preserve,
                           )


def _aggregate_numpy(data, func, group_columns, top_n, sort_by, round_floats=3, 
                     preserve=['author', 'author_id']):
    """
    Group a dataframe by the column names listed in group_columns. Then for every numeric variable in 
    data, compute the numpy operation specified by func. Numeric variables that need to be preserved
    (such as author_id or eid) can be passed by preserve. Return the resulting DataFrame. 

    Parameters
    ----------
    data: pd.DataFrame
        Data from which to aggregate
    func: str
        Which numpy function to use in aggregation. 
    group_columns: list
        List of column names in data by which to group the results. These must be 
        valid column names in data
    top_n: int
        The top n originators (by paper count) which to include in the output. 
    sort_by: str
        Column name by which to sort to get the top values if top_n is not None 
    round_floats: bool
        How many values to which to round the results of the aggregation. Perform no
        rounding by setting this value to None.
    preserve: list
        List of column names that should be preserved during aggregation. The first value
        seen for these columns in each aggregation group is used. An example use case 
        may be seen when group_columns contains 'author_id'. If 'author' is not in 
        preserve, only 'author_id' will determine the identity of the author in the 
        aggregated results. Since each 'author' and 'author_id' pair should be unique,
        using the first seen author name for an arbitrarty author id will accurately 
        preserve this information. Note that columns in preserve should be distinct 
        from columns in group_columns.
        
    Returns
    -------
    aggregated_data: pd.DataFrame
        The aggregated results from data
    """
    # make sure columns in `preserve` exist in group_columns
    preserve_set = set(preserve)
    columns_set = set(data.columns)
    preserve = preserve_set & columns_set
    if preserve != preserve_set:
        warnings.warn('One or more entries in `preserve` does not exist as a column in DataFrame')
    preserve = list(preserve)
    
    # remove any values common to both group_columns and preserve from preserve
    # values in group_columns are indeces in the grouped data, preserved by default
    if set(group_columns) & set(preserve):
        for g in group_columns:
            if g in preserve:
                preserve.remove(g)

    # build a custom query to aggregate numeric columns by some numpy function 'func'
    # numeric columns in preserve are not included in this aggregation and the first 
    # encountered value for each respective column is chosen. 
    numeric_columns = set(data.select_dtypes(include=[np.number]).columns)
    numeric_columns -= set(preserve)
    query = f"data.groupby({group_columns}).apply(lambda x: pd.Series({{"
    for col in preserve:
        query += f"'{col}': x.iloc[0]['{col}'], "
    
    for col in numeric_columns:
        if col in group_columns:
            continue
        
        if round_floats is not None:
            condition = f"'{col}': round(np.{func}(x.{col}), {round_floats}), "
        else:
            condition = f"'{col}': np.{func}(x.{col}), "
        query += condition

    query += "})).reset_index()"
    formatted_df = eval(query)  # evaluate the string query, producing aggregated dataframe
    
    # select the top_n samples if requested
    if top_n is not None:
        if sort_by is None:
            raise ValueError('top_n selected but no value given to sort_by')
    
        formatted_df = formatted_df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
        formatted_df = formatted_df.iloc[:top_n]
    return formatted_df