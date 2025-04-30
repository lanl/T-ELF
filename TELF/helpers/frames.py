import pandas as pd
import numpy as np
import rapidfuzz
import uuid
import warnings

def create_coauthor_affil_df(auth_affil_map, name_map):
    data = {
        'Author ID': [],
        'Author Name': [],
        'Co-Author ID': [],
        'Co-Author Name': [],
        'Affiliation ID': [],
        'Affiliation Name': [],
        'Country': [],
        'Year': [],
        'Num. Shared Publications': [],
    }

    for auth_id, aff_auth in auth_affil_map.items():
        for aff_id, aff_year in aff_auth.items():
            for year, aff_info in aff_year.items():
                for c_auth_id, aff_c_auth in aff_info['coauthors'].items():
                    for c_aff_id, c_aff_info in aff_c_auth.items():
                        data['Author ID'].append(auth_id)
                        data['Author Name'].append(name_map.get(auth_id, 'Unknown'))
                        data['Co-Author ID'].append(c_auth_id)
                        data['Co-Author Name'].append(name_map.get(c_auth_id, 'Unknown'))
                        data['Affiliation ID'].append(c_aff_id)
                        data['Affiliation Name'].append(c_aff_info['name'])
                        data['Country'].append(c_aff_info['country'])
                        data['Year'].append(year)
                        data['Num. Shared Publications'].append(c_aff_info['num_shared_publications'])

    coauth_affil_df = pd.DataFrame.from_dict(data)
    return coauth_affil_df

def create_author_affil_df(auth_affil_map, name_map):
    data = {
        'Author ID': [],
        'Author Name': [],
        'Affiliation ID': [],
        'Affiliation Name': [],
        'Country': [],
        'Year': [],
        'Num. Publications': [],
        'Num. Citations': [],
        'Num. Coauthors': [],
    }

    for auth_id, aff_auth in auth_affil_map.items():
        for aff_id, aff_year in aff_auth.items():
            for year, aff_info in aff_year.items():
                data['Author ID'].append(auth_id)
                data['Author Name'].append(name_map.get(auth_id, 'Unknown'))
                data['Affiliation ID'].append(aff_id)
                data['Affiliation Name'].append(aff_info['name'])
                data['Country'].append(aff_info['country'])
                data['Year'].append(year)
                data['Num. Publications'].append(aff_info['num_publications'])
                data['Num. Citations'].append(aff_info['num_citations'])
                data['Num. Coauthors'].append(len(aff_info['coauthors']))

    auth_affil_df = pd.DataFrame.from_dict(data)
    return auth_affil_df


def reorder_and_add_columns(df, order, fill_value=np.nan):
    """
    Reorders the columns of the dataframe based on the provided column order list.
    Adds columns that do not exist in the original dataframe with the specified fill value.
    Columns in the dataframe that are not in the `order` will be placed at the end.

    Parameters:
    -----------
    df: pd.DataFrame
        The input dataframe.
    order: list
        The desired column order including non-existing columns.
    fill_value: Any
        The value to fill in for the non-existing columns. Default is NaN.

    Returns:
    --------
    pd.DataFrame: 
        The dataframe with columns reordered and non-existing columns added.
    """
    # Ensure all columns in `order` are included in the result, adding missing ones
    columns_to_add = [col for col in order if col not in df.columns]

    # Add missing columns to `df` with the specified fill value
    for col in columns_to_add:
        df[col] = fill_value

    # Determine the final column order
    final_order = [col for col in order if col in df.columns] + \
                  [col for col in df.columns if col not in order]

    # Reorder columns in `df` according to `final_order`
    return df[final_order]


def drop_duplicates(df, col):
    """
    Drop duplicates from a DataFrame based on a specified column while preserving rows with NaN values 
    in that column. Among duplicates, rows with fewer NaN values in other columns are prioritized.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The input DataFrame from which duplicates are to be removed.
    col: str
        The name of the column based on which duplicates should be identified and dropped.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with duplicates removed 
    """
    df_nan_col = df[df[col].isna()]  # separate out rows where the column has NaN
    df = df.dropna(subset=[col]).copy()
    df['nan_count'] = df.isna().sum(axis=1)  # add helper column for nan counts
    df = df.sort_values(by=[col, 'nan_count'])  # sort by col and then by nan_count
    
    # drop duplicates (prioritizing papers with fewer NaNs due to sorting)
    df = df.drop_duplicates(subset=col, keep='first')
    df = df.drop(columns='nan_count')
    df = pd.concat([df, df_nan_col], axis=0).reset_index(drop=True)
    return df

def match_frames_text(df1, df2, col, min_len=10, threshold=0.05):
    """
    Matches rows from two dataframes based on text similarity in a specified column.
    
    Parameters:
    -----------
    df1: pd.DataFrame
        The first dataframe.
    df2: pd.DataFrame
        The second dataframe.
    col: str
        The column name on which to perform text similarity matching.
    min_len: int 
        The minimum length of a title to be matched. 
    threshold: float
        Minimum normalized Indel similarity score for matching entries. This value should be on [0, 1). 
        The smaller the `threshold`, the better the text match. Default is 0.05 which roughly 
        corresponds to a 95% match based on normalized Indel similarity. 

    Returns:
    --------
    tuple
        Two dataframes with a new column indicating matched keys.
    """
    match_col = f'{col}_key'
    df1[match_col] = [None] * len(df1)  # init empty columns
    df2[match_col] = [None] * len(df2)

    for idx1, row1 in df1.iterrows():
        if pd.isna(row1[col]):
            continue
        elif len(row1[col]) < min_len:
            continue
        extract = rapidfuzz.process.extractOne(row1[col], df2[col], score_cutoff=threshold, 
                                               scorer=rapidfuzz.distance.Indel.normalized_distance)
        if extract is None:
            continue
        else:
            match, score, idx2 = extract
            match_key = str(uuid.uuid4())
            df1.at[idx1, match_col] = match_key
            df2.at[idx2, match_col] = match_key
            
    return df1, df2, match_col

def merge_frames_simple(df1, df2, key):
    """
    Merges two dataframes on a specified key. The common columns between the dataframes (excluding the key) 
    are identified and combined, with preference given to df1.
    
    Parameters:
    -----------
    df1: pd.DataFrame
        The first dataframe.
    df2: pd.DataFrame
        The second dataframe.
    key: str
        The column name to merge the dataframes on. This column should be present in both dataframes.
    
    Returns:
    --------
    pd.DataFrame: 
        A merged dataframe with combined columns from both dataframes
    """
    # find common columns excluding the key
    common = [col for col in df1.columns if col in df2.columns and col != key]
    
    # merge frames
    merged_df = df1.merge(df2, on=key, how='outer', suffixes=('_df1', '_df2'))
    
    # combine common columns prioritizing df1
    for col in common:
        with warnings.catch_warnings():
            # TODO: pandas >= 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
            warnings.filterwarnings("ignore", category=FutureWarning)
            merged_df[col] = merged_df[col + '_df1'].combine_first(merged_df[col + '_df2'])
        merged_df.drop(columns=[col + '_df1', col + '_df2'], inplace=True)
    return merged_df

def merge_frames(df1, df2, col, common_cols, key=None, remove_duplicates=False, 
                 remove_nan=False, order=None):
    """
    Merges two dataframes based on a specified column and combines values of common columns.
    
    This function takes two dataframes `df1` and `df2`, and merges them based on the specified 
    column `col`. For the columns listed in `common_cols`, it combines values, prioritizing 
    non-NaN values from `df1`. The resulting dataframe can optionally be filtered and reordered.

    Parameters:
    -----------
    df1: pd.DataFrame
        Primary DataFrame.
    df2: pd.DataFrame
        Secondary DataFrame
    col: str
        The column name based on which the two dataframes should be merged.
    common_cols: list[str]
        List of column names that exist in both dataframes and should be combined.
    key: str, optional
        Column name based on which rows with NaN values should be dropped and duplicates removed.
        If None, this is not done. Default is None.
    remove_duplicates: bool
        If key is set, remove the duplicates in the column associated with this key
    remove_nan: bool
        If key is set, remove the nans in the column associated with this key
    order: list[str], optional
        List of column names specifying the desired order of columns in the resulting dataframe.
    
    Returns:
    --------
    pd.DataFrame: 
        Merged DataFrame
    """
    df1[col] = df1[col].str.lower()
    df2[col] = df2[col].str.lower()

    merged_df = df1.merge(df2, on=col, how='outer', suffixes=('_df1', '_df2'))

    # use common_cols to combine the columns and drop the temporary ones
    for common_col in common_cols:
        col_df1 = common_col + '_df1'
        col_df2 = common_col + '_df2'
        
        merged_df[common_col] = merged_df[col_df1].combine_first(merged_df[col_df2])
        merged_df = merged_df.drop(columns=[col_df1, col_df2])

    if key is not None:  # remove nans and duplicates (optional)
        if remove_nan:
            merged_df = merged_df.dropna(subset=[key])
        if remove_duplicates:
            merged_df = drop_duplicates(merged_df, key)
    if order is not None:  # reset order (optional)
        merged_df = merged_df[order].reset_index(drop=True)

    return merged_df