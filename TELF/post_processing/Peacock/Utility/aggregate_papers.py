import ast
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from .util import filter_by, fuzzy_duplicate_match

# known duplicate affiliation IDs (SCOPUS affiliation ids)
DEFINED_AFFILIATIONS = {
    'LANL': {'60014338','60006164','60006099','60020912','60004377','60026138','60017069','60014338','60029965','121906248','60020912','113822884','113822613'},
    'LLNL': {'60026175','60121635'},
    'SNL': {'60007843','60015900','60014682','60204096','60025871','60121851'},
    'INL': {'60005120'},
    'BNL': {'60006221','60025119','60029842','60009683','60007472'},
    'ANL': {'60028609'},
    'ORNL': {'60024266','60003029','60030241','60026875','60003954','60003577'},
    'PNNL': {'60023471','60074893','60074766'}
}

DEFAULT_COL_NAMES = {
    'id': 'eid',
    'year': 'year',                     
    'funding': 'funding',
    'authors': 'authors',
    'author_ids': 'author_ids',
    'affiliations': 'affiliations',
    'citations': 'outCitations',
    'references': 'inCitations'
 }


def per_funding_dataframe(df, common=None, col_names=None):
    """ convert dataframe where primary key is paper to a data frame with primary key funding """

    # setup column names
    if col_names is None:
        col_names = {}
    tmp = DEFAULT_COL_NAMES.copy()
    tmp.update(col_names)
    col_names = tmp

    id_col = col_names['id']
    year_col = col_names['year']
    funding_col = col_names['funding']
    citations_col = col_names['citations']

    # drop any necessary null values from the original dataframe
    subset=[funding_col, year_col, id_col]
    if common:
        subset += common
    funding_df = df.dropna(subset=subset)

    # compute a 2D list of funding information from whole data
    funding_list = [ast.literal_eval(f) for f in funding_df[funding_col].to_list()]

    # 1. use str comparison to group any similar organization names
    fnames = {x for f in funding_list for x in f} # set of all unique funding orgs
    matches = fuzzy_duplicate_match(fnames)

    # 2. get counts for each funding organization. also determine country
    org_country_map = {}
    org_counts = {}
    for fobj in funding_list:
        for org, org_details in fobj.items():
            org_country = org_details['funding_country']
            if org not in org_country_map:
                org_country_map[org] = org_country
            if org not in org_counts:
                org_counts[org] = 0
            org_counts[org] += 1

    # 3. create map for consolidated organizations
    joint_org_map = {}
    slic_id_map = {}
    slic_count = 0
    for orgs in matches:
        if len(orgs) == 1:
            org = next(iter(orgs))
            joint_org_map[org] = org
            slic_id_map[org] = f'f{slic_count}'
            slic_count += 1
        else:
            most_common = max(orgs, key=lambda x: org_counts[x])
            slic_tmp = 1
            for org in orgs:
                if org_country_map[org] == org_country_map[most_common]:
                    joint_org_map[org] = most_common
                    slic_id_map[org] = f'f{slic_count}'
                else:
                    joint_org_map[org] = org
                    slic_id_map[org] = f'f{slic_count + slic_tmp}'
                    slic_tmp += 1
            slic_count += slic_tmp

    # count how many organizations comprise the joint organization
    joint_counts = dict(Counter(joint_org_map.values()))

    # 4. generate the funding dataframe
    data = {
        'funding_id': [],
        'funding': [],
        'country': [],
        'num_funding_orgs': [],
        'num_citations': [],  # number of citations (NOT references)
        'year': [],
        'id': [],
    }

    # init data columns for specified common variables
    if common is not None:
        if not isinstance(common, list):
            common = [common]
        for c in common:
            data[c] = []


    for funding, year, eid, oc in zip(funding_df[funding_col].to_list(), 
                                      funding_df[year_col].to_list(), 
                                      funding_df[id_col].to_list(),
                                      funding_df[citations_col].to_list()):

        # process the citations column
        if pd.isna(oc):
            num_oc = 0
        elif isinstance(oc, str) and ';' in oc:
            num_oc = len(oc.split(';'))
        else:
            num_oc = int(oc)
        
        funding = ast.literal_eval(funding)
        for org in funding:
            joint_org = joint_org_map[org]
            joint_org_id = slic_id_map[org]
            num_orgs = joint_counts[joint_org]
            org_country = org_country_map[joint_org]

            data['year'].append(year)
            data['funding_id'].append(joint_org_id)
            data['funding'].append(joint_org)
            data['country'].append(org_country)
            data['num_funding_orgs'].append(num_orgs)
            data['num_citations'].append(num_oc)   
            data['id'].append(eid)
            
    if common:  # update data with common variables
        for c in common:

            # create map of paper id to common variable
            c_map = {k:v for k,v in zip(funding_df[id_col].to_list(), funding_df[c].to_list())}
            for eid in data['id']:  # fill data using map
                data[c].append(c_map.get(eid))
    
    return pd.DataFrame.from_dict(data)


def per_author_dataframe(df, use_author_map=True, use_affil_map=True, common=None, col_names=None):

    # setup column names
    if col_names is None:
        col_names = {}
    tmp = DEFAULT_COL_NAMES.copy()
    tmp.update(col_names)
    col_names = tmp

    authors_col = col_names['authors']
    author_ids_col = col_names['author_ids']
    affiliations_col = col_names['affiliations']

    # set up author map
    if use_author_map:
        authID_to_name = {}
        for name_list, id_list in zip(df[authors_col].to_list(), df[author_ids_col].to_list()):
            if pd.notna(name_list) and pd.notna(id_list):  # check for non-NaN values before processing
                for name, aid in zip(name_list.split(';'), id_list.split(';')):
                    if pd.notna(name) and pd.notna(aid):  # ensure both name and ID are defined
                        authID_to_name[aid] = name
    else:
        authID_to_name = None


    # set up affiliation id to affiliation name map
    if use_affil_map:
        affID_to_name = {}

        # reverse defined affiliations map
        defined_affiliations = {x: k for k, v in DEFINED_AFFILIATIONS.items() for x in v}

        for aff_list in df[affiliations_col].to_list():
            if pd.notna(aff_list):
                try:
                    aff_dict = ast.literal_eval(aff_list)
                    if isinstance(aff_dict, dict):  # check if dict
                        for aff_id, aff_data in aff_dict.items():
                            if 'name' in aff_data and pd.notna(aff_data['name']):
                                name = aff_data['name']
                            else:
                                name = 'Unknown'
                            if aff_id not in affID_to_name or affID_to_name[aff_id] == 'Unknown':
                                affID_to_name[aff_id] = defined_affiliations.get(aff_id, name)
                except (ValueError, SyntaxError):
                    print(f"Error parsing affiliation data: {aff_list}")
        
        # update map with reverse defined affiliations, overwriting any 'Unknown' with defined names
        affID_to_name.update(defined_affiliations)
    else:
        affID_to_name = None
        
    auth_df = per_author_dataframe_helper(df, authID_to_name, affID_to_name, common, col_names)
    return auth_df


def per_author_dataframe_helper(df, author_map=None, affil_map=None, common=None, col_names=None):
    """ convert dataframe where primary key is paper to a data frame with primary key author """
    
    # setup column names
    if col_names is None:
        col_names = {}
    tmp = DEFAULT_COL_NAMES.copy()
    tmp.update(col_names)
    col_names = tmp

    id_col = col_names['id']
    year_col = col_names['year']
    citations_col = col_names['citations']
    author_ids_col = col_names['author_ids']
    affiliations_col = col_names['affiliations']

    data = {
        'author_id': [],
        'author': [],
        'year': [],
        'affiliation_id': [],
        'affiliation': [],
        'country': [],
        'num_citations': [],  # number of citations (NOT references)
        'id': [],
    }

    # init data columns for specified common variables
    if common is not None:
        if not isinstance(common, list):
            common = [common]
        for c in common:
            data[c] = []

    # if no map, set to as empty map instead of NoneType
    if author_map is None:
        author_map = {}
    if affil_map is None:
        affil_map = {}
    
    # reverse the defined affiliations map
    defined_affiliations = {x:k for k,v in DEFINED_AFFILIATIONS.items() for x in v}  # TODO: remove global
    
    # make sure no null values exist for target columns
    subset=[author_ids_col, year_col, id_col, affiliations_col]
    if common:
        subset += common
    if 'index' in subset:
        subset.remove('index')  # remove index column since it cant be NaN
        df['index'] = df.index  # copy index to column for key-word access
    auth_df = df.dropna(subset=subset)
    for authorID_list, year, eid, aff, oc in zip(auth_df[author_ids_col].to_list(), 
                                                      auth_df[year_col].to_list(), 
                                                      auth_df[id_col].to_list(),
                                                      auth_df[affiliations_col].to_list(),
                                                      auth_df[citations_col].to_list()):
        
        aff = ast.literal_eval(aff)
        temp_aff = {}
        temp_country = {}
        for k,v in aff.items():
            for a in v['authors']:
                temp_country[a] = v['country']
                if k in defined_affiliations:
                    temp_aff[a] = next(iter(DEFINED_AFFILIATIONS[defined_affiliations[k]]))
                else:
                    temp_aff[a] = k
        
        # process the citations column
        if pd.isna(oc):
            num_oc = 0
        elif isinstance(oc, str) and not oc.isnumeric():
            num_oc = len(oc.split(';'))
        else:
            num_oc = int(oc)
        
        for auth_id in authorID_list.split(';'):            
            data['author_id'].append(auth_id)
            data['author'].append(author_map.get(auth_id, None))
            data['year'].append(year)
            data['affiliation_id'].append(temp_aff.get(auth_id, None))  # some authors not represnted in aff structure if not valid affid
            data['affiliation'].append(affil_map.get(temp_aff.get(auth_id, 'Unknown'), None))
            data['country'].append(temp_country.get(auth_id,None))
            data['num_citations'].append(num_oc)   
            data['id'].append(eid)
            

    if common:  # update data with common variables
        for c in common:

            # create map of paper id to common variable
            c_map = {k:v for k,v in zip(auth_df[id_col].to_list(), auth_df[c].to_list())}
            for eid in data['id']:  # fill data using map
                data[c].append(c_map.get(eid))
    
    return pd.DataFrame.from_dict(data)


def generate_map(data, uid, col):
    """
    Given two columns in a DataFrame, generate a map of one column to the other
    column. Note that, these columns should have a unique underlying mapping (ie. 
    author id and author name columns) for sensical results. 


    Parameters
    ----------
    data: pd.DataFrame
        data from which to create map
    uid: str
        column in data the values of which will serve as keys in the map
    col: str
        column in data the values of which will serve as values in the map

    Returns
    -------
    uid_map: dict
        the computed map
    """
    uid_map = {}
    uids = set(data[uid].unique())
    for u in tqdm(uids):
        uid_data = data.loc[data[uid] == u]
        uid_map[u] = uid_data[col].mode()[0]
    return uid_map


def generate_originator_statistics(data, key, top_n=10, sort_by='paper_count', by_year=True, filters=None, col_names=None):
    """
    Given a dataframe where the primary key is author/author_id, generate a DataFrame
    of statistics on the originator (this can be author, affiliation, or country). 
    The originator is specified by the 'key' argument. The output dataframe will contain
    three statistics: paper_count, attribution_percentage, and num_citations. Paper_count 
    corresponds to how many papers the originator is responsible for. Attribution_percentage
    represents (on average) how much inividual work the originator provides on their papers. 
    An attribution_percentage of 1 means that only the originator is featured on their papers.
    An attribution_percentage of 0.25 means that, on average, each paper published by the
    originator has three other originators. Num_citations is the total sum of citations 
    accumulated from papers published by the originator.


    Parameters
    ----------
    data: pd.DataFrame
        Papers DataFrame where author/author_id is the primary key. 
        In practice this will always be output from per_author_dataframe()
    key : str
        The originator on which to base the statistics. Valid options are
        'author_id', 'affiliation_id', 'country'.
    top_n: int
        The top n originators (by paper count) which to include in the output. 
        Default is 10.
    sort_by: str
        Column name by which to sort to get the top values if top_n is not None
        Default is None
    by_year: bool
        Flag that determines whether to generate statistics per year or globally.
        Default is True.

    Returns
    -------
    selected_data: pd.DataFrame
        output DataFrame with statistics
    """

    ### 0. choose what data is preserved depending on whether author_id, affiliation_id, funding_id, or country is passed
    if key == 'author_id':
        keys = ['author_id', 'author', 'affiliation_id', 'affiliation', 'country']
    elif key == 'affiliation_id':
        keys = ['affiliation_id', 'affiliation', 'country']
    elif key == 'funding_id':
        if 'country' in data:
            keys = ['funding_id', 'funding', 'country'] 
        else:
            keys = ['funding_id', 'funding'] 
    elif key == 'country':
        keys = ['country']
    else:
        raise ValueError(f'"{key}" is a not a valid key. Options are ["author_id", "affiliation_id", "funding_id", "country"')
    
    ### 1. compute paper contributions for originator per year
    # first step in generating attribution percentage for each year, for each author/affiliation
    # this section of code computes the authors/affiliations associated with each paper
    paper_contributions = {}
    selected_data = data.copy()
    for index, row in selected_data.iterrows():
        pid = row.id
        key_instance = row[key]  # get the author/affiliation id at the current row

        if pid not in paper_contributions:
            paper_contributions[pid] = {key_instance}
        else:
            paper_contributions[pid].add(key_instance)
    
    ### 2. if sort_by is paper count or number of citations, we can select top_n here 
    # otherwise we need to compute attribution percentage and then select top_n
    if sort_by in {'paper_count', 'num_citations'}:
        if filters is not None:
            selected_data = filter_by(selected_data, filters)

        if sort_by == 'paper_count':
            top = set(selected_data.groupby([key])['id'].nunique().sort_values(ascending=False).iloc[:top_n].index)
        else:
            top = set(selected_data.drop_duplicates(['id', 'year', key]).groupby(key).agg(
                num_citations = ('num_citations', 'sum')
            ).sort_values(by='num_citations', ascending=False).iloc[:top_n].index)
        selected_data = selected_data.loc[selected_data[key].isin(top)].copy()
    
    ### 3. calculate map of papers published by top authors in a given year
    # second step in calculating attribution percentage
    # for each author/affiliation id in a given year, get their papers
    # then divide number of their papers by number of total contributors for those papers
    key_to_papers = {}
    for index, row in selected_data.iterrows():
        pid = row.id
        year = row.year
        key_id = row[key]

        # create a subdictionary for the author/affiliation 
        if key_id not in key_to_papers:
            key_to_papers[key_id] = {}
        key_subdict = key_to_papers[key_id]

        # populate year with author ids
        if year not in key_subdict:
            key_subdict[year] = {pid}
        else:
            key_subdict[year].add(pid)

    # for each author/affiliatin, for each year, compute the attribution percentage
    key_contributions = {}
    for key_id, key_subdict in key_to_papers.items():
        if key_id not in key_contributions:
                key_contributions[key_id] = {}
        for year, papers in key_subdict.items():
            key_contributions[key_id][year] = np.mean([1 / len(paper_contributions[p]) for p in papers])
            
    # add attribution percentage to dataframe
    attribution_percentage = []
    for year, key_id in zip(selected_data['year'].to_list(), selected_data[key].to_list()):
        key_subdict = key_contributions.get(key_id, {})
        ap = key_subdict.get(year, None)
        attribution_percentage.append(ap)
    selected_data['attribution_percentage'] = attribution_percentage

    
    ### 4A. If using author_id as key, make sure to set affiliation to most frequent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore pandas future warning on sort functinality
        if key == 'author_id':
            selected_data['country'] = selected_data.groupby(key)['country'].transform(lambda x: x.mode(dropna=False)[0])
            selected_data['affiliation'] = selected_data.groupby(key)['affiliation'].transform(lambda x: x.mode(dropna=False)[0])
            selected_data['affiliation_id'] = selected_data.groupby(key)['affiliation_id'].transform(lambda x: x.mode(dropna=False)[0])
        elif key == 'affiliation_id':
            selected_data['country'] = selected_data.groupby(key)['country'].transform(lambda x: x.mode(dropna=False)[0])
        elif key == 'funding_id' and 'country' in keys:
            selected_data['country'] = selected_data.groupby(key)['country'].transform(lambda x: x.mode(dropna=False)[0])

    ### 4B. aggregate the data
    # need to aggregrate number of citations and unique papers separately to avoid duplicate count
    # first get the number of citations
    tmp_a = selected_data.drop_duplicates(['id', 'year', key]).groupby(keys+['year']).agg(
        num_citations = ('num_citations', 'sum')
    ).reset_index()
    
    # now get the number of papers and attribution percentage
    tmp_b = selected_data.groupby(keys+['year']).agg(
        paper_count = ('id', pd.Series.nunique),
        attribution_percentage = ('attribution_percentage', lambda x: x.iloc[0]),
    ).reset_index()
    
    # merge these two dataframes to get the data for plotting
    selected_data = tmp_b.merge(tmp_a, on=keys+['year']).reset_index(drop=True)
    
    ### 4C. select by attribution percentage if requested
    if sort_by == 'attribution_percentage':
        tmp = aggregate_ostats_time(selected_data, key)
        if filters is not None:
            tmp = filter_by(tmp, filters)
        top = set(tmp.sort_values(by=sort_by, ascending=False).iloc[:top_n][key])
        selected_data = selected_data.loc[selected_data[key].isin(top)].copy()

    ### 5. aggregate across time (if requested)
    if by_year:
        return selected_data
    else:
        return aggregate_ostats_time(selected_data, key)


def aggregate_ostats(data, key, common=None, top_n=10, sort_by='paper_count', col_names=None, filters=None, by_year=True):
    """
    Go from a standard papers dataframe where paper id is the primary key to a
    dataframe of statistics on the originator. The originator is either author,
    affiliation, or country. 

    Parameters
    ----------
    data: pd.DataFrame
        Data from which to aggregate
    common: list
        List of data columns names, the values of which are the same when data is ordered
        by paper id or by author id. For example, a 'cluster' column would be common as 
        papers belong to a cluster no matter how data is ordered. 
    key: str
        List of column names in data by which to group the results. These must be 
        valid column names in data
    top_n: int
        The top n originators (by paper count) which to include in the output. 
        Default is 10.
    sort_by: str
        Column name by which to sort to get the top values if top_n is not None
        Default is None
    by_year: bool
        Flag that determines whether to generate statistics per year or globally.
        Default is True.

    Returns
    -------
    stat_df: pd.DataFrame
        The aggregated statistics from data
    """
    if key == 'funding_id':
        formatted_df = per_funding_dataframe(data, common=common, col_names=col_names)
    else:
        formatted_df = per_author_dataframe(data, common=common, col_names=col_names)
    stat_df = generate_originator_statistics(formatted_df, top_n=top_n, key=key, sort_by=sort_by, by_year=by_year, filters=filters, col_names=col_names)
    return stat_df


def aggregate_ostats_time(stat_df, key):
    """ 
    For an originator statistics dataframe split by time, aggregate data over that time.
    Note that the aggregated attribution percentage will be a weighted mean
    

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame generated by generate_originator_statistics()
    key: str
        The originator on which to base the statistics. Valid options are
        'author_id', 'affiliation_id', 'country'.
    sort_by: str
        Which statistic to sort the aggregated results by

    Returns
    -------
    stat_df: pd.DataFrame
        time-aggregated originator statistics DataFrame 
    """
    
    # choose what data is preserved depending on whether author_id, affiliation_id, or country is passed
    if key == 'author_id':
        keys = ['author_id', 'author', 'affiliation_id', 'affiliation', 'country']
    elif key == 'affiliation_id':
        keys = ['affiliation_id', 'affiliation', 'country']
    elif key == 'funding_id':
        if 'country' in stat_df.columns:
            keys = ['funding_id', 'funding', 'country'] 
        else:
            keys = ['funding_id', 'funding'] 
    elif key == 'country':
        keys = ['country']
    else:
        raise ValueError(f'"{key}" is not a valid key. Options: ["author_id", "affiliation_id", "funding_id", "country"]')

    # Drop duplicates before grouping to avoid conflict on insert
    grouped = stat_df.groupby(keys, as_index=False).agg({
        'paper_count': 'sum',
        'num_citations': 'sum',
        'attribution_percentage': lambda x: np.average(x, weights=stat_df.loc[x.index, 'paper_count'])
    })

    return grouped



def count_countries(df, col='affiliations'):
    """
    Given a papers DataFrame with affiliation information, produce a dictionary of country counts
    
    Parameters:
    -----------
    data: pd.DataFrame
        Data from which to count countries
    col: str, (optional)
        The column which contains affiliation information (including country). Expected to be the 
        normal SLIC affiliation dict format. Default is 'affiliations'.
        
    Returns:
    --------
    dict:
        Dict where keys are countries and values are int count
    """
    affiliations = [ast.literal_eval(x) for x in df[col].to_list() if isinstance(x, str)]
    countries = []
    for aff_info in affiliations:
        for aff_id, aff in aff_info.items():
            countries.append(aff['country'])
    return dict(Counter(countries))


def count_affiliations(df, col='affiliations'):
    """
    Given a papers DataFrame with affiliation information, produce a dictionary of affiliation counts.
    Since one affiliation id can have multiple affiliation names (ex: Los Alamos, LANL, Los Alamos Lab, etc), 
    the most common name for any given affiliation id is chosen as the key in the output.
    
    Parameters:
    -----------
    data: pd.DataFrame
        Data from which to count affiliations
    col: str, (optional)
        The column which contains affiliation information. Expected to be the normal SLIC affiliation dict format. 
        Default is 'affiliations'.
        
    Returns:
    --------
    dict:
        Dict where keys are affiliation names and values are int count
    """
    output = []
    names = {}
    affiliations = [ast.literal_eval(x) for x in df[col].to_list() if isinstance(x, str)]
    for aff_info in affiliations:
        for aff_id, aff in aff_info.items():
            if aff_id not in names:
                names[aff_id] = []
            
            names[aff_id].append(aff['name'])
            output.append(aff_id)
    
    output = dict(Counter(output))
    combined = {}
    for key, count in output.items():
        most_common_name = Counter(names[key]).most_common(1)[0][0]
        combined[most_common_name] = count
    return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))