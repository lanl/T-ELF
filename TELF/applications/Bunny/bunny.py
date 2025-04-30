import os
import re
import ast
import sys
import random
import warnings
import pandas as pd
from typing import Union, List
from collections import Counter
from dataclasses import dataclass
pd.set_option('future.no_silent_downcasting', True)

from ..Penguin import Penguin
from ..Penguin.crocodile import form_df
from ...pre_processing.iPenguin.Scopus import Scopus
from ...pre_processing.iPenguin.SemanticScholar import SemanticScholar
from ...pre_processing.iPenguin.utils import format_pubyear
from ...pre_processing.Orca.AuthorMatcher import AuthorMatcher

@dataclass
class BunnyFilter:
    filter_type: str
    filter_value: str

    def __post_init__(self):
        if self.filter_type not in Bunny.FILTERS and self.filter_type != 'DOI':
            raise ValueError(f'Invalid filter_type "{self.filter_type}". Allowed values are {list(Bunny.FILTERS.keys())}')


@dataclass
class BunnyOperation:
    operator: str
    operands: List[Union[BunnyFilter, 'BunnyOperation']]

    def __post_init__(self):
        if self.operator not in {"AND", "OR"}:
            raise ValueError(f'Invalid operator "{self.operator}". Only "AND" and "OR" are allowed.')

        for operand in self.operands:
            if not isinstance(operand, (BunnyFilter, BunnyOperation)):
                raise ValueError(f"Invalid operand {operand}. Operands must be instances of Filter or Operation.")


def is_valid_query(query):
    """
    Validates the structure of a Bunny filter query.

    The function checks whether a given input follows the intended structure of Bunny filter
    queries. Each element in the query is evaluated recursively, making sure that only BunnyFilter
    and BunnyOperation objects are present. 

    Parameters
    ----------
    query: (Filter/Operation)
        The query to be validated. The query can be an instance of BunnyFilter or BunnyOperation dataclass.

    Returns
    -------
    bool: 
        True if the query is valid, False otherwise.
    """
    if isinstance(query, BunnyFilter):
        if query.filter_type is None or query.filter_value is None:
            return False
    elif isinstance(query, BunnyOperation):
        if query.operator is None or query.operands is None:
            return False
        for operand in query.operands:  # check each operand recursively
            if not is_valid_query(operand):
                return False
    else:  # if part of a query is not a BunnyFilter or BunnyOperation, it's invalid
        return False
    return True


def evaluate_query_to_string(query):
    """
    Convert a Bunny query into a string that can be accepted by Scopus.
    
    Parameters
    ----------
    query: BunnyFilter, BunnyOperation
        The query object. This can be an instance of BunnyFilter or BunnyOperation.
    
    Returns
    -------
    str
        A string representation of the query that can be processed by Scopus
    """
    
    # If the query is a Filter, just return its string representation
    if isinstance(query, BunnyFilter):
        if query.filter_type == 'PUBYEAR':
            year, fmat = format_pubyear(query.filter_value)
            if year is None:
                raise ValueError(f'"{query.filter_value}" is an invalid PUBYEAR. Options are ["YEAR", "-YEAR", "YEAR-"]')
            else:
                return f"{query.filter_type} {fmat} {year}"
        else:
            return f"{query.filter_type}({query.filter_value})"

    # If the query is an Operation, recursively evaluate each operand
    elif isinstance(query, BunnyOperation):
        subquery_strings = [evaluate_query_to_string(subquery) for subquery in query.operands]
        return f"({f' {query.operator} '.join(subquery_strings)})"
    
    else:
        raise ValueError(f"Invalid query type: {type(query)}")


def find_doi(f):
    """
    Helper function that attempts to extract the DOI from a Scopus paper XML file
    
    Parameters
    ----------
    f: str
        Path to the Scopus XML file
    
    Returns
    -------
    doi: str, None
        Returns DOI if found, else None
    """
    with open(f, 'rb') as fh:  # read only first 2048 bytes
        data = fh.read(1536)
        while True:  # if splitting multibyte character, try to read less
            try:
                data = data.decode('utf-8')
                break
            except UnicodeDecodeError:
                data = data[:-1]

    pattern = '<prism:doi>(.*?)</prism:doi>'
    match = re.search(pattern, data, re.DOTALL)
    return match.group(1) if match else None


class Bunny():
    
    MODES = {'references', 'citations', 's2_author_ids'}
    FILTERS = {'AFFILCOUNTRY': 'Country', 
               'AFFILORG': 'Affiliation', 
               'AF-ID': 'Affiliation ID',
               'PUBYEAR': 'Publication Year',
               'AU-ID': 'Scopus Author ID',
               'KEY': 'Keyword',}
    
    def __init__(self, s2_key=None, scopus_keys=None, penguin_settings=None, output_dir='.', verbose=False):
        self.output_dir = output_dir
        self.verbose = verbose
        self.s2_key = s2_key
        self.scopus_keys = scopus_keys
        self.penguin_settings = penguin_settings
        self.enabled = self.s2_key is not None

        # Explicitly map supported filters to methods
        self.filter_funcs = {
            'AFFILCOUNTRY': lambda df, f, auth_map=None: self._filter_affil_generic(df, column_name='country', filter_value=f, auth_map=auth_map),
            'AFFILORG': self._filter_affilorg,
            'AF-ID': self._filter_afid,
            'PUBYEAR': self._filter_pubyear,
            'AU-ID': self._filter_auid,
            'KEY': self._filter_key,
            'DOI': lambda df, f, auth_map=None: set(df[df['doi'].str.lower() == f.lower()].index),
        }

    
    def __init_lookup(self, series, priority, sep):
        lookup = [y for x in series for y in x.split(sep)]
        lookup_freq = Counter(lookup)
        unique_lookup = list(lookup_freq.keys())

        # sort based on priority
        if priority == 'random':
            random.shuffle(unique_lookup)
        elif priority == 'frequency':
            unique_lookup.sort(key=lambda x: lookup_freq[x], reverse=True)
        else:
            raise ValueError("`priority` should be in ['random', 'frequency']")
        return unique_lookup
        
    
    def __hop(self, df, col, hop, hop_focus, s2_dir, sep, max_papers, hop_priority):
        # setup access to Penguin cache
        ignore = None
        if self.penguin_settings is not None:
            ignore = self.get_penguin_cache(source='s2')
        
        # setup ipenguin object
        ip = SemanticScholar(key=self.s2_key, mode='fs', name=os.path.join(self.output_dir, s2_dir), 
                             ignore=ignore, verbose=self.verbose)
        
        # drop the papers that were not matched by either doi or title
        tmp_df = df.dropna(subset=[col])
        tmp_df = tmp_df.loc[tmp_df.type == hop]
        lookup = self.__init_lookup(tmp_df[col].to_list(), hop_priority, sep)
        
        if hop_focus is not None:
            lookup = list(set(lookup) & set(hop_focus))
        if not lookup:
            raise RuntimeError('Nothing to lookup!')
        
        # make the hop
        if col in ('citations', 'references'):
            hop_df, targets = ip.search(lookup, mode='paper', n=max_papers)
        elif col in ('s2_author_ids'):
            hop_df, targets = ip.search(lookup, mode='author', n=max_papers)
        else:
            hop_df, targets = None, None

        if targets is not None and self.penguin_settings is not None:
            penguin = Penguin(**self.penguin_settings)
            penguin.add_many_documents(os.path.join(self.output_dir, s2_dir), source='s2', overwrite=True)
            hop_df = penguin.id_search(
                ids = [f's2id:{x}' for x in targets],
                as_pandas=True
            )
        return hop_df


    @classmethod
    def estimate_hop(cls, df, col='citations'):
        """
        Predict the maximum number of papers that will be in the DataFrame if another hop is performed. 
        The numbers of papers in the next hop is guaranteed to be less than or equal to this number

        Parameters
        ----------
        df: pandas.DataFrame
            The target Bunny DataFrame

        Returns
        -------
        int
            Maximum possible number of papers contained in the next hop
        """
        hop_ids = (set.union(*[set(x.split(';')) for x in df[col] if not pd.isna(x)]))
        current_ids = {x for x in df.s2id.to_list() if not pd.isna(x)}
        return len(hop_ids | current_ids)


    @classmethod
    def suggest_filter_values(cls, df, options=['AFFILCOUNTRY', 'AFFILORG']):
        """
        Generate the possible Bunny filtering values for some options.
        Note that these values do not conclusively map all possible filter values used by Scopus
        but instead produce all filter values currently used by the given DataFrame df
        
        Parameters
        ----------
        df: pandas.DataFrame
            The target Bunny DataFrame

        Returns
        -------
        suggestions: dict
            A dictionary where the keys are the filter types and the values are a set of suggested filter
            values
        """
        supported_options = ('AFFILCOUNTRY', 'AFFILORG')
        for opt in options:
            if opt not in supported_options:
                raise ValueError(f'Unknown option "{opt}" provided!. Supported options include {supported_options}')

        suggestions = {}
        for opt in options:
            suggestions[opt] = set()
        if 'affiliations' in df:
            for aff in df.affiliations.to_list():
                if pd.isna(aff) or aff is None or aff == 'None' or aff == 'nan':
                    continue
    
                org_set = set()
                country_set = set()
                if isinstance(aff, str):
                    aff = ast.literal_eval(aff)
                for aff_id in aff:
                    org_set.add(aff[aff_id]['name'])
                    country_set.add(aff[aff_id]['country'])
                if 'AFFILORG' in suggestions:
                    suggestions['AFFILORG'] |= org_set
                if 'AFFILCOUNTRY' in suggestions:
                    suggestions['AFFILCOUNTRY'] |= country_set
        return suggestions

    
    def form_core_scopus(self, data, data_type, keys, s2_dir='s2', scopus_dir='scopus'):
        if data_type not in ('scopus_author_id'):
            raise ValueError(f'Invalid data type: "{data_type}"')
        
        scopus = Scopus(
            keys = keys,
            mode = 'fs',
            name = os.path.join(self.output_dir, scopus_dir),
            verbose=self.verbose
        )
        
        query = BunnyOperation('OR', [BunnyFilter('AU-ID', x) for x in data])
        query_str = evaluate_query_to_string(query)
        
        # get the scopus dataframe
        scopus_df = scopus.search(query_str, n=0)
        
        # drop papers with missing authors, affiliations
        scopus_df.dropna(subset=['author_ids', 'affiliations', 'doi'], inplace=True)
        scopus_df['type'] = [0] * len(scopus_df)
        
        # extract dois
        dois = scopus_df.doi.to_list()

        # get s2df
        s2 = SemanticScholar(
            key = self.s2_key, 
            mode ='fs', 
            name = os.path.join(self.output_dir, s2_dir), 
            verbose=self.verbose
        )
        
        dois = [f'DOI:{x}' for x in dois]
        s2_df = s2.search(dois, mode='paper')        
        s2_df['type'] = [0] * len(s2_df)
        
        # add scopus info back in
        s2_df.dropna(subset=['doi'], inplace=True)
        s2_df.doi = s2_df.doi.str.lower()
        s2_dois = set(s2_df.doi.to_list())
        
        scopus_df.doi = scopus_df.doi.str.lower()
        scopus_df = scopus_df.loc[scopus_df.doi.isin(s2_dois)].copy()
        return form_df(s2_df, scopus_df)

    
    def form_core(self, data, data_type, s2_dir='s2'):
        if not self.enabled:
            raise RuntimeError('Bunny object was not initialized with an S2 API key.')
        #if data_type not in ('doi', 's2id', 'bibtex', 's2_author_id'):
        if data_type not in ('paper', 'author'):
            raise ValueError(f'Invalid data type: "{data_type}"')
        
        # setup ipenguin object
        ip = SemanticScholar(key=self.s2_key, mode='fs', name=os.path.join(self.output_dir, s2_dir), 
                             verbose=self.verbose)
        
        if data_type == 'paper':
            core_df, _ = ip.search(data, mode='paper')
        elif data_type == 'author':
            core_df, _ = ip.search(data, mode='author')
        
        if core_df.empty:  # no results for given input paper uids
            return None
        else:
            core_df['type'] = [0] * len(core_df)  # add a type column to the core df
            return core_df
    
    
    def hop(self, df, hops, modes, use_scopus=False, filters=None, hop_focus=None, scopus_keys=None, 
            s2_dir='s2', scopus_dir='scopus', filter_in_core=True, max_papers=0, hop_priority='random'):
        """
        Perform one or more hops along the citation network

        This function allows the user to perform one or more hops on the citation network, using either 
        references or citations as the method of expansion. The user can also optionally specify filters
        (using BunnyFilter format) to limit the scope of the search to some area. Note that if filters are
        being used, the user must specify Scopus API keys as only the Scopus API can provide filtering
        in the Bunny implementation.

        Parameters
        ----------
        df: pandas.DataFrame
            The target Bunny DataFrame
        hops: int
            How many hops to perform. Must be >= 1
        modes: (list, set, tuple)
            Which mode(s) to use Bunny in. Can either be 'citations', 'references', 's2_author_ids'
        use_scopus: boolean
            Flag that determines if Scopus API should be called to fill in more detailed information such as affiliation,
            country of origin, keywords (PACs), etc. scopus_keys must be provided if this flag is set to True.
        filters: BunnyFilter or BunnyQuery
            Specialized dataclass used for representing a boolean query at any level of nesting. See example notebooks
            to see how these filters can be created. use_scopus must be True and scopus_keys need to be provided to use
            Bunny filters. Default=None. 
        scopus_keys: list
            List of Scopus API keys which are used to call on the Scopus API to enrich hop-expanded DataFrame. Default=None.
        scopus_dir: str, Path
            The directory at which to create a Scopus paper archive. This will cache previosly downloaded papers to save time
            and API call limits. The directory is expectd to exist inside Bunny.output_dir and a new directory will be created
            at Bunny.output_dir/scopus_dir if it cannot be found. Default=scopus.
        filter_in_core: bool
            Flag that determines whether any filters should be applied to the core. This option is only needed if filters are
            specified. If True, core papers can be filtered and removed from the Bunny DataFrame. Default=True.
        scopus_batch_size: int
            The maximum batch size for looking up papers using the Scopus API. Note that Scopus sets a maximum Boolean query 
            at 10,000. The size of the Boolean query can be calculated approximately using (num filters + 1) * scopus_batch_size.
            Care should be taken to decrease the batch size if many filters are being used. 
        max_papers: int
            This variable is used to set an upper limit on how many papers can be featured in a hop. If set to 0, no upper
            limit will be used. Default is 0.
        hop_priority: str
            If `max_papers` is not 0, this variable is used to prioritize which papers are selected for the hop. Options are 
            in ['random', 'frequency']. Default is 'random'.

        Returns
        -------
        pd.DataFrame
            Hop-expanded result DataFrame
        """
        if not self.enabled:
            raise RuntimeError('Bunny object was not initialized with an S2 API key.')
        if isinstance(modes, str):
            modes = [modes]
        for m in modes:
            if m not in self.MODES:
                raise ValueError(f'Invalid expansion mode "{m}" selected for hop. Valid modes include: {self.MODES}.')
        if not isinstance(hops, int) or hops <= 0:
            raise ValueError('Invalid value for hops "{hops}". This argument must be an int >= 1.')
        if use_scopus and scopus_keys is None:
            raise ValueError('scopus_keys must be provided when use_scopus=True.')
        if scopus_keys and filters:
            warnings.warn('[Bunny]: scopus_keys and filters were provided but ' 
                          'use_scopus was set to False. Setting use_scopus=True.')
            use_scopus=True
        
        # clean hop column
        df['type'] = df['type'].fillna(0)
        df.type = df.type.astype(int)
        
        # establish how many hops to make and where to start
        first_hop = max(df.type) + 1

        if hop_focus is None:
            hop_focus = {k: None for k in modes}
        
        # make the hops 
        for i in range(first_hop, hops+first_hop):
            for m in modes:
                if self.verbose:
                    print(f'[Bunny]: Downloading papers for hop {i}', file=sys.stderr)
                hop_df = self.__hop(df, col=m, hop=i-1, hop_focus=hop_focus[m], s2_dir=s2_dir, 
                                    sep=';', max_papers=max_papers, hop_priority=hop_priority)
                if hop_df is not None:
                    hop_df['type'] = [i] * len(hop_df)
                    df = pd.concat([df, hop_df], axis=0).reset_index(drop=True)
                else:
                    print(f"Warning: hop_df is None for i = {i}")
            
        s2_df = df.drop_duplicates(subset=['s2id'], keep='first').reset_index(drop=True)
        if use_scopus:  # get scopus info / filter
            return self.get_affiliations(s2_df, scopus_keys, filters, scopus_dir, filter_in_core, True)
        return s2_df


    def __downloaded_scopus(self, dois, name, keys, filters):
        ignore = None
        if self.penguin_settings is not None:
            ignore = self.get_penguin_cache(source='scopus')
        
        scopus = Scopus(
            keys = keys,
            mode = 'fs',
            name = os.path.join(self.output_dir, name),
            ignore = ignore,
            verbose = self.verbose
        )
        
        query = BunnyOperation('OR', [BunnyFilter('DOI', x) for x in dois])
        if filters is not None:
            if not is_valid_query(filters):  # validate the Bunny filter query 
                raise ValueError('The provided Bunny filters could not be validated.')
            query = BunnyOperation('AND', [query, filters])

        query_str = evaluate_query_to_string(query)
        df, targets = scopus.search(query_str, n=0)
        if self.penguin_settings is not None:
            penguin = Penguin(**self.penguin_settings)
            penguin.add_many_documents(os.path.join(self.output_dir, name), source='scopus', overwrite=True)
            df = penguin.id_search(
                ids = [f'eid:{x}' for x in targets],
                as_pandas=True
            )
        return df
    

    def __verify_s2_df(self, df):
        """
        Determine if the provided DataFrame matches the structure expected of a Bunny Semantic Scholar
        DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame in question

        Returns
        -------
        bool
            True if valid. False otherwise.
        """
        return True


    def __verify_scopus_df(self, df):
        """
        Determine if the provided DataFrame matches the structure expected of a Bunny Scopus
        DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame in question

        Returns
        -------
        bool
            True if valid. False otherwise.
        """
        return True


    def get_affiliations(self, df, scopus_keys, filters, save_dir='scopus', filter_in_core=True, do_author_match=True):
        df_order = [
            'eid', 
            's2id', 
            'doi', 
            'title', 
            'abstract',
            'year',
            'authors',
            'author_ids', 
            'affiliations', 
            'funding',
            'PACs', 
            'publication_name',
            'subject_areas',
            's2_authors',
            's2_author_ids',
            'citations',
            'num_citations',
            'references',
            'num_references',
            'type',
        ]
        
        if 'eid' in df:  # DataFrame already has column(s) for Scopus information
            # make sure that df matches the format of Bunny with Scopus info
            if not self.__verify_scopus_df(df):
                raise ValueError('Input DataFrame does not match expected format!')
            
            # split data into data with scopus information and data without scopus information
            s2_df = df.loc[(df.eid.isnull()) & (~df.doi.isnull())]
            scopus_df = df.drop(s2_df.index)

            # lookup papers on scopus by doi
            dois = s2_df.doi.to_list()
            if dois:
                processed_df = self.__downloaded_scopus(dois, save_dir, scopus_keys, filters)
                if processed_df is not None and not processed_df.empty :
                    if self.verbose:
                        print(f'[Bunny]: Joining S2 and Scopus DataFrames. . .', file=sys.stderr)
                    processed_df = form_df(s2_df, processed_df, df_order)
                    joined_df = pd.concat([scopus_df, processed_df], ignore_index=True)
                else:
                    joined_df = df
            else:
                joined_df = df

        else:  # no scopus information is present
            # make sure that df matches the format of Bunny with S2 only info
            if not self.__verify_s2_df(df):
                raise ValueError('Input DataFrame does not match expected format!')
            
            tmp_df = df.dropna(subset=['doi'])
            dois = tmp_df.doi.to_list()
            processed_df = self.__downloaded_scopus(dois, save_dir, scopus_keys, filters)
            joined_df = form_df(df, processed_df, df_order)


        if filters is not None:
            if self.verbose:
                print(f'[Bunny]: Applying filters. . .', file=sys.stderr)
            return self.__filter(joined_df, filters, filter_in_core, do_author_match)
        return joined_df.loc[~joined_df.s2id.isnull()]
    
    
    def _filter_pubyear(self, df, f, auth_map=None):
        if 'year' not in df:
            raise ValueError('"year" not found in df')

        pids = set()
        year, fmat = format_pubyear(f)
        if year is None:
            raise ValueError(f'Unexpected format for PUBYEAR: "{year}"')
        
        year_df = df.dropna(subset=['year'])
        if fmat == 'IS':
            pids = set(df.loc[df.year == year].index.to_list())
        elif fmat == 'AFT':
            pids = set(df.loc[df.year > year].index.to_list())
        else:
            pids = set(df.loc[df.year < year].index.to_list())
        return pids
    

    def _filter_key(self, df, f, auth_map=None):
        keyword = f
        cols = ['title', 'abstract', 'PACs']
        mask = pd.Series(False, index=df.index)
        for col in cols:
            if col in df.columns:
                mask |= df[col].str.contains(r'\b' + re.escape(keyword) + r'\b', case=False, regex=True)

        pids = set(mask[mask].index)
        return pids


    def _filter_auid(self, df, f, auth_map=None):
        mask = pd.Series(False, index=df.index)
        pids = set(mask.index)
        return pids

    
    def _filter_affil_generic(self, df, column_name, filter_value, auth_map): 
        if 'affiliations' not in df:
            raise ValueError('"affiliations" not found in df')

        filter_value = filter_value.lower()
        pids, aids = set(), set()
        aff_df = df.dropna(subset=['affiliations'])
        affiliations = {k: v for k, v in zip(aff_df.index.to_list(), aff_df.affiliations.to_list())}

        for idx, affiliation in affiliations.items():
            if isinstance(affiliation, str):
                affiliation = ast.literal_eval(affiliation)
            for aff_id, aff in affiliation.items():
                try:
                    if aff[column_name].lower() == filter_value:
                        pids.add(idx)
                        aids |= set(aff['authors'])
                        break
                except KeyError:
                    print(f"Warning: '{column_name}' not found in affiliation {aff_id} for index {idx}.")
                except Exception as e:
                    print(f"Warning: error processing affiliation {aff_id} at index {idx} — {e}")

        if auth_map is not None:
            s2_aids = {auth_map[aid] for aid in aids if aid in auth_map}
            for idx, scopus_authors, s2_authors in zip(df.index.to_list(), df.author_ids.to_list(), df.s2_author_ids.to_list()):
                if isinstance(scopus_authors, str) and set(scopus_authors.split(';')) & aids:
                    pids.add(idx)
                if isinstance(s2_authors, str) and set(s2_authors.split(';')) & s2_aids:
                    pids.add(idx)    

        return pids


    

    def _filter_affilorg(self, df, f, auth_map):
        if 'affiliations' not in df:
            raise ValueError('"affiliations" not found in df')

        org = f.lower()
        pids, aids = set(), set()
        aff_df = df.dropna(subset=['affiliations'])
        affiliations = {k:v for k,v in zip(aff_df.index.to_list(), aff_df.affiliations.to_list())}
        for idx, affiliation in affiliations.items():
            if isinstance(affiliation, str):
                affiliation = ast.literal_eval(affiliation)
            for aff_id, aff in affiliation.items():
                if aff['name'].lower() == org:
                    pids.add(idx)
                    aids |= set(aff['authors'])
                    break
                    
        if auth_map is not None:
            s2_aids = {auth_map[aid] for aid in aids if aid in auth_map}
            for idx, scopus_authors, s2_authors in zip(df.index.to_list(), df.author_ids.to_list(), df.s2_author_ids.to_list()):
                if isinstance(scopus_authors, str) and set(scopus_authors.split(';')) & aids:
                    pids.add(idx)
                if isinstance(s2_authors, str) and set(s2_authors.split(';')) & s2_aids:
                    pids.add(idx)   
        return pids
        

    def _filter_afid(self, df, f, auth_map):
        if 'affiliations' not in df:
            raise ValueError('"affiliations" not found in df')

        afid = f.lower()
        pids, aids = set(), set()
        aff_df = df.dropna(subset=['affiliations'])
        affiliations = {k:v for k,v in zip(aff_df.index.to_list(), aff_df.affiliations.to_list())}
        for idx, affiliation in affiliations.items():
            if isinstance(affiliation, str):
                affiliation = ast.literal_eval(affiliation)
            for aff_id, aff in affiliation.items():
                if aff_id.lower() == afid:
                    pids.add(idx)
                    aids |= set(aff['authors'])
                    break
        
        if auth_map is not None:
            s2_aids = {auth_map[aid] for aid in aids if aid in auth_map}
            for idx, scopus_authors, s2_authors in zip(df.index.to_list(), df.author_ids.to_list(), df.s2_author_ids.to_list()):
                if isinstance(scopus_authors, str) and set(scopus_authors.split(';')) & aids:
                    pids.add(idx)
                if isinstance(s2_authors, str) and set(s2_authors.split(';')) & s2_aids:
                    pids.add(idx)    
        return pids
    

    def __filter(self, df, filters, filter_in_core, do_author_match):
        all_pids = []
        if do_author_match and {'s2id', 'eid', 's2_author_ids', 's2_authors', 'authors', 'author_ids'}.issubset(df.columns):
            if self.verbose:
                print('[Bunny]: Using Orca to match Scopus author IDs to s2 author IDs.')
            am = AuthorMatcher(df, verbose=self.verbose)
            auth_match_df = am.match()
            auth_map = {scopus_auth: s2_auth for scopus_auth, s2_auth in zip(auth_match_df['SCOPUS_Author_ID'].to_list(), auth_match_df['S2_Author_ID'].to_list())}
        else:
            auth_map = None
        
        # validate the Bunny filter query 
        if not is_valid_query(filters):
            raise ValueError('The provided Bunny filters could not be vaidated.')

        pids = self.__evaluate_query(filters, df, auth_map)
        if not filter_in_core:
            pids |= set(df.loc[df.type == 0].index.to_list())
        filtered_df = df.loc[df.index.isin(pids)]
        
        if 's2id' in df.columns:
            return filtered_df.dropna(subset=['s2id'])
        else: 
            return filtered_df

    def __evaluate_query(self, query, df, auth_map):
        if isinstance(query, BunnyFilter):
            ffunc = self.filter_funcs[query.filter_type]
            result = ffunc(df, query.filter_value, auth_map)
            return result
        elif isinstance(query, BunnyOperation):
            results = [self.__evaluate_query(operand, df, auth_map) for operand in query.operands]
            if query.operator == 'AND':
                return set.intersection(*results)
            elif query.operator == 'OR':
                return set.union(*results)
    
    def apply_filter(self, df, filters, filter_in_core=True, do_author_match=True):
        if 'eid' not in df and do_author_match:
            do_author_match = False
            warnings.warn('Input DataFrame does not contain Scopus information', RuntimeWarning)

        filtered_df = self.__filter(df, filters, filter_in_core, do_author_match)
        return filtered_df


    def add_to_penguin(self, source, path):
        """
        Get the downloaded papers for a given `source` collection from Penguin, represented as a 
        fixed memory-size Bloom filter. This can then be given to iPenguin child objects to prevent
        re-downloading previously acquired papers.
        
        Parameters:
        -----------
        source: str
            The name of the source collection from which to retrieve IDs. It should correspond to
            either SCOPUS_COL or S2_COL.
        path: str, pathlike
            The path to the directory where the downloaded files are cached, ready to be added into 
            Penguin.
        """
        if self.penguin_settings is None:
            raise ValueEror('Tried to use Penguin when connection details were never provided')
        penguin = Penguin(**self.penguin_settings)
        penguin.add_many_documents(path, source=source, overwrite=True)
    
    
    def get_penguin_cache(self, source, max_items=1.25, false_positive_rate=0.001):
        """
        Get the downloaded papers for a given `source` collection from Penguin, represented as a 
        fixed memory-size Bloom filter. This can then be given to iPenguin child objects to prevent
        re-downloading previously acquired papers. For more details on the Bloom filter, see 
        Penguin.get_id_bloom.
        
        Parameters:
        -----------
        source: str
            The name of the source collection from which to retrieve IDs. It should correspond to
            either SCOPUS_COL or S2_COL.
        max_items: float, int, (optional)
            The maximum number of items expected to be stored in the Bloom filter. This can be a
            fixed integer or a float representing a multiplier of the current document count in the
            collection. Default is 1.25.
        false_positive_rate: float, (optional)
            The desired false positive probability for the Bloom filter. Default is 0.001. 
    
        Returns:
        --------
        rbloom.Bloom
            An instance of a Bloom filter populated with IDs from the specified collection.
    
        Raises:
        -------
        ValueError
            If 'source' does not match any of the predefined collection names, or if 'max_items'
            is not a float or an int.

            If attempting to use this function without Penguin connection string details provided
        """
        if self.penguin_settings is None:
            raise ValueEror('Tried to use Penguin when connection details were never provided')
        penguin = Penguin(**self.penguin_settings)
        return penguin.get_id_bloom(source=source,
                                    max_items=max_items,
                                    false_positive_rate=false_positive_rate)
    
    ### Setters / Getters


    @property
    def s2_key(self):
        return self._s2_key

    @s2_key.setter
    def s2_key(self, s2_key):
        if s2_key is not None:
            try:
                ip = SemanticScholar(key=s2_key)
            except ValueError:
                raise ValueError(f'The key "{s2_key}" was rejected by the Semantic Scholar API')
        self._s2_key = s2_key

    @property
    def penguin_settings(self):
        return self._penguin_settings

    @penguin_settings.setter
    def penguin_settings(self, penguin_settings):
        if penguin_settings is not None:
            if not isinstance(penguin_settings, dict):
                raise TypeError(f'Unsupported type {type(penguin_settings)} for `penguin_settings`. Expected dict')

            if 'uri' not in penguin_settings:
                raise ValueError('Key "uri" is required to be passed in penguin_settings')
            if 'db_name' not in penguin_settings:
                raise ValueError('Key "db_name" is required to be passed in penguin_settings')

            # set username and password to None if not passed
            if 'username' not in penguin_settings:
                penguin_settings['username'] = None
            if 'password' not in penguin_settings:
                penguin_settings['password'] = None
        self._penguin_settings = penguin_settings
