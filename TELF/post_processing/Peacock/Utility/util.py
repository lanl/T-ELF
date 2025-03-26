import textwrap
from functools import lru_cache
import rapidfuzz


def filter_by(data, filters):
    """
    Filter a DataFrame based on a set of criteria defined in the `filters` dictionary. 
    Each key-value pair in `filters` specifies a column and the values to filter by in that column.

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to filter.
    filters: dict
        A dictionary where each key is a column name in `data` and each value is a value or list of values
        to filter by in the specified column. The function filters the data to include only rows where 
        the column specified by the key contains one of the values in the corresponding list.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the rows that meet all the filtering criteria specified in `filters`.

    Raises
    ------
    TypeError
        If `filters` is not a dictionary.
    """
    if not isinstance(filters, dict): 
        raise TypeError('filter_by should be a dictionary of filters (keys) and targets (values)')

    selected_data = data.copy()
    for f, targets in filters.items():
        if not isinstance(targets, list):
            targets = [targets]
        targets = set(targets)
        selected_data = selected_data.loc[selected_data[f].isin(targets)].copy()
    return selected_data



def create_label_column(data, col_a, col_b=None, max_len=30):
    """
    Adds a 'labels' column to the DataFrame, creating labels based on the contents of one or two other columns. 
    If two columns are provided, their values are concatenated with a hyphen as a separator. Optionally, the 
    labels can be wrapped to a specified maximum length.

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame to which the 'labels' column will be added.
    col_a: str
        The name of the first column to use in creating the labels. This column's values will always be 
        included in the labels.
    col_b: str, (optional)
        The name of the second column to use in creating the labels. If provided, its values will be 
        concatenated to those of `col_a` with a hyphen separator. If `None`, only `col_a` is used for labels.
    max_len: int, (optional)
        The maximum length of each label. If the concatenated label exceeds this length, it will be wrapped 
        to a new line at the nearest whitespace character that does not exceed the maximum length. 
        Default is 30 characters.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with an added 'labels' column containing the generated labels.
    """

    # set the labels depending on whether author_id, affiliation_id, or country is passed
    if col_b is not None:
        data['labels'] = [f'{a} - {b}' for a, b in zip(data[col_a], data[col_b])]
    else:
        data['labels'] = data[col_a].to_list()

    # add new lines to labels
    data['labels'] = [textwrap.fill(x, width=max_len) for x in data['labels'].to_list()]
    return data


def lev_dist(a, b):
    """Calculates the levenshtein distance between two input strings a and b"""
    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):

        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),      # insert character
            min_dist(s1 + 1, s2),      # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )
    return min_dist(0, 0)


def str_compare(a, b, thresh=0.075):
    lenn = max(len(a), len(b))
    dist = lev_dist(a, b)
    return dist / lenn <= thresh


def fuzzy_duplicate_match(strings, thresh=95):
    seen = set()
    matches = []
    for k1 in strings:
        if k1 in seen:
            continue

        match = {k1}
        for k2 in strings:
            if rapidfuzz.fuzz.ratio(k1, k2, score_cutoff=thresh):
                match.add(k2)

        seen |= match
        matches.append(match)
    return matches

def save_fig(fig, fname, interactive):
    # save or return the figure
    if fname:
        if interactive:
            if not fname.endswith('.html'):  # save as interactive html
                fname += '.html'
            fig.write_html(fname)
        else:
            if not fname.endswith('.png'):  # save as static png
                fname += '.png'
            fig.write_image(fname, scale=4)
    else:
        return fig

    
def create_label_annotate_text(row, cols):
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(cols, list):
        return format_val(row[cols[0]])
    else:
        raise TypeError('Unknown type for `labels`. This parameter should be a string or list of strings of DataFrame columns')

def create_label_hover_text(row, cols):
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(cols, list):
        out = []
        for col in cols:
            val = row[col]
            fmat_val = format_val(val)
            out.append(f'<b>{col}</b>: {fmat_val}')
        return '<br>'.join(out)
    else:
        raise TypeError('Unknown type for `labels`. This parameter should be a string or list of strings of DataFrame columns')
        

def format_val(val, digits=3):
    """
    Formats the given numerical value to a specified number of decimal places or returns it unchanged.

    This function checks if the input `val` is a float. If `val` is a float and is an integer (e.g., 2.0),
    it is converted to an integer type. If `val` is a float but not an integer, it is formatted to the specified number 
    of decimal places, given by the `digits` parameter. If `val` is not a float, it is returned unchanged.

    Parameters:
    -----------
    val: int, float, any
        The value to be formatted.
    digits: int, (optional)
        The number of decimal places to round `val` if it is a float. Default is 3.

    Returns:
    --------
    int, str, any
        The formatted value
    """
    if isinstance(val, float):
        if val.is_integer():
            return int(val)
        else:
            return f'{val:.3f}'
    else:
        return val
        
        
def process_label_cols(cols):
    """
    Processes the input to ensure it is a list of strings representing DataFrame column names.

    This function takes an input `cols` which can be a single string or a list of strings.
    It converts a single string into a list containing that string. It also validates that, if a list is provided,
    it consists only of strings. If `cols` is None or a data structure the bool of which is False then an empty
    list is returned. Any other input will raise an error.

    Parameters:
    -----------
    cols: str, list of str, None
        The input which is either a single column name as a string, a list of column names, or None.

    Returns:
    --------
    list
        A validated list of column names. Note that this function does not check for membership in the DataFrame.
        Only the structure of the data is validated.

    Raises:
    -------
    ValueError
        If `cols` is a list containing non-string elements.
    TypeError
        If `cols` is neither a string, a list of strings, nor None.
    """
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(cols, list):
        if any(not isinstance(x, str) for x in cols):
            raise ValueError('`labels` must be composed of a list of strings!')
    elif not cols:
        cols = []
    else:
        raise TypeError('Unknown type for `labels`. This parameter should be a string or list of strings of DataFrame columns')
    return cols   