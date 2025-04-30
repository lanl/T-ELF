import random
import warnings
import rapidfuzz
import pandas as pd
from tqdm import tqdm
import multiprocessing
from collections import Counter
from itertools import permutations
from joblib import Parallel, delayed
from ....helpers.data_structures import gen_chunks
        
        
def text_compare(S2_pidToAids, S2_aidToName, authIdToS2pid, authName, authId, sub_table=None, depth=None, threshold=0.8):
    s2_pids = authIdToS2pid[authId][:depth]  # get a list of s2 paper ids that are written by the author
    consensus = []
    for pid in s2_pids:
        if pid in S2_pidToAids:
            aid_set = S2_pidToAids[pid]
            
            curr = (None, 0)
            for aid in aid_set:
                if aid in S2_aidToName:
                    scopus_name = authName
                    s2_name = S2_aidToName[aid]
                    score = match_name(scopus_name, s2_name, sub_table, threshold)
                    if score > curr[1] and score >= threshold:
                        curr = (aid, score)
            
            if curr[0] is not None:
                consensus.append(curr)
        
    if consensus:
        return set(consensus)
    else:
        return None
    
    
def get_common_authors(S2_pidToAids, authIdToS2pid, authId, depth=11):
    '''
    Find a set of S2 author ids that are all listed as authors on papers
    written by author with scopus authId. Could be an empty set.
    '''
    try:
        if depth:  # get a list of s2 paper ids that are written by the author
            s2_pids = authIdToS2pid[authId][:depth]
        else:
            s2_pids = authIdToS2pid[authId]
    except KeyError:
        return {}, 0
    
    while True:
        try:
            s2_id_list = [S2_pidToAids[x] for x in s2_pids]
            break
        except KeyError as e:  # s2 paper id not found in map
            s2_pids.remove(e.args[0])  # remove paper not found in s2 map
    
    if not s2_id_list or not s2_id_list[0]:
        return set(), 0
    else:
        counts = Counter([x for y in s2_id_list for x in y])
        try:
            max_count = counts.most_common(1)[0][1]
        except IndexError:
            print(s2_id_list, counts)
            raise Exception
        matched_authors = {value for value, count in counts.most_common() if count == max_count}
        return matched_authors, max_count
    
    
def match_authors(auth_ids, s2_id_map, s2_name_map, authid_to_s2, authid_to_name, sub_table=None, verbose=False):
    errors = []
    auth_map = {}
    count = 0
    for auth_id in tqdm(auth_ids, disable = not verbose):
        auth_name = authid_to_name[auth_id]
        common_auths, freq = get_common_authors(s2_id_map, authid_to_s2, auth_id, depth=None)
        if not common_auths:
            errors.append(auth_id)
        # elif len(common_auths) == 1:
        #     s2_authId = next(iter(common_auths))
        #     if s2_authId not in auth_map:
        #         auth_map[s2_authId] = (auth_id, freq)
        #     else:
        #         if auth_map[s2_authId][1] < freq:
        #             errors.append(auth_map[s2_authId][0])
        #             auth_map[s2_authId] = (auth_id, freq)
        #         else:
        #             errors.append(auth_id)
        else:
            s2_authIds = text_compare(s2_id_map, s2_name_map, authid_to_s2, auth_name, auth_id, sub_table, depth=None, threshold=0.8)
            if s2_authIds is not None:
                for s2_authId,_ in s2_authIds:
                    if s2_authId and s2_authId not in auth_map:
                        auth_map[s2_authId] = (auth_id, 0)
                        
            else:
                count += 1
                
    return auth_map, errors, count


def normalize_string(s, sub_table):
    sub_table = str.maketrans(sub_table)
    return s.translate(sub_table)


def generate_variations(name):
    tokens = name.split()
    variations = []

    # for each permutation of the tokens, combine one token with its abbreviation and keep the other token as it is
    for perm in permutations(tokens, 2):
        long_name, short_name = perm
        abbrev = short_name[0] + "."
        variations.append(f"{long_name} {abbrev}".lower())

    # remove duplicates
    variations = list(set(variations))
    return variations
    
    
def match_name(scopus_name, s2_name, sub_table=None, threshold=0):

    # normalize the scopus and s2 name characters
    if sub_table is not None:
        sub_table = str.maketrans(sub_table)
        s2_name = s2_name.translate(sub_table)
        scopus_name = scopus_name.translate(sub_table)
    
    # adjust scopus name
    split = scopus_name.split()
    ln = split[0]
    fn = split[-1]
    fn = fn.split('.')[0]
    adjusted_name = f'{ln} {fn}.'.lower()
    name_variations = generate_variations(s2_name)

    fuzz_output = rapidfuzz.process.extractOne(adjusted_name, name_variations, score_cutoff=threshold, scorer=rapidfuzz.distance.Indel.normalized_similarity)
    if fuzz_output is None:
        return 0
    else:
        _, score, _ = fuzz_output
        return score

    
def error_match(errors, auth_map, s2_id_map, s2_name_map, authid_to_s2, authid_to_name, sub_table):
    error_map = {}
    for auth_id in errors:
        auth_name = authid_to_name[auth_id]
        s2_authIds = text_compare(s2_id_map, s2_name_map, authid_to_s2, auth_name, auth_id, sub_table=sub_table, depth=None)
        if s2_authIds:
            for s2_authId,_ in s2_authIds: 
                if s2_authId and s2_authId not in auth_map:
                    error_map[s2_authId] = (auth_id, 0)

    return error_map
        
    
class AuthorMatcher:
    
    # dictionary of common characters that can be normalized
    CHAR_NORMALIZATION_DICT = {  
        'Á': 'A', 'á': 'a',
        'À': 'A', 'à': 'a',
        'Ä': 'A', 'ä': 'a',
        'Â': 'A', 'â': 'a',
        'Å': 'A', 'å': 'a',
        'Ã': 'A', 'ã': 'a',
        'Ā': 'A', 'ā': 'a',
        'Ą': 'A', 'ą': 'a',
        'Ă': 'A', 'ă': 'a',

        'Ç': 'C', 'ç': 'c',

        'É': 'E', 'é': 'e',
        'È': 'E', 'è': 'e',
        'Ê': 'E', 'ê': 'e',
        'Ë': 'E', 'ë': 'e',
        'Ę': 'E', 'ę': 'e',
        'Ē': 'E', 'ē': 'e',

        'Í': 'I', 'í': 'i',
        'Ì': 'I', 'ì': 'i',
        'Î': 'I', 'î': 'i',
        'Ï': 'I', 'ï': 'i',
        'İ': 'I', 'ı': 'i',

        'Ł': 'L', 'ł': 'l',

        'Ñ': 'N', 'ñ': 'n',
        'Ń': 'N', 'ń': 'n',
        'Ň': 'N', 'ň': 'n',

        'Ó': 'O', 'ó': 'o',
        'Ò': 'O', 'ò': 'o',
        'Ö': 'O', 'ö': 'o',
        'Ô': 'O', 'ô': 'o',
        'Õ': 'O', 'õ': 'o',
        'Ō': 'O', 'ō': 'o',
        'Ø': 'O', 'ø': 'o',

        'Ŕ': 'R', 'ŕ': 'r',

        'Š': 'S', 'š': 's',
        'Ş': 'S', 'ş': 's',
        'Ș': 'S', 'ș': 's',
        
        'Ť': 'T', 'ť': 't',
        'Ţ': 'T', 'ţ': 't',
        
        'Ú': 'U', 'ú': 'u',
        'Ù': 'U', 'ù': 'u',
        'Ü': 'U', 'ü': 'u',
        'Û': 'U', 'û': 'u',
        'Ū': 'U', 'ū': 'u',
        'Ů': 'U', 'ů': 'u',

        'Ý': 'Y', 'ý': 'y',
        'Ÿ': 'Y', 'ÿ': 'y',

        'Ž': 'Z', 'ž': 'z',
        'Ź': 'Z', 'ź': 'z',
        'Ż': 'Z', 'ż': 'z',
    }
        
    def __init__(self, df, n_jobs=-1, verbose=False):
        """
        Parameters
        ----------
        df: str, pd.DataFrame
            The input papers DataFrame intended to be used for correspondence between authors from Semantic Scholar and Scopus
            datasets. If df is a string, it is assumed to be the path where the DataFrame is saved on disk. 
        n_jobs: int, optional
            Number of parallel processes. The default is -1.
        verbose: bool, optional
            Verbosity. The default is False
        """
        if isinstance(df, str):
            df = pd.read_csv(df)
        
        self.df = df
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        
    def match(self, a_id='eid', a_authors=('authors', 'author_ids'), 
                    b_id='s2id', b_authors=('s2_authors', 's2_author_ids'), 
                    delimiter=';', known_matches={}):
        """
        Correspond the author ids between Scopus and Semantic Scholar
        
        Parameters
        ----------
        a_id: str, optional
            The column name for the paper identifier of dataset A.
            Default is the Scopus default.
        a_authors: (str, str), optional
            Tuple of columns names for author names and authors ids for dataset A.
            Default is the Scopus default.
        b_id: str, optional
            The column name for the paper identifier of dataset B
            Default='s2id' (The Semantic Scholar default)
        b_authors: (str, str), optional
            Tuple of columns names for author names and authors ids for dataset B.
            Default is the Semantic Scholar default.
        delimiter: str, optional
            The character by which to delimit the author names and ids. The default is ';'.
        """
        if self.verbose:
            print(f'[Orca]: Matching Scopus Authors to Semantic Scholar Authors')
        
        a_auth_names, a_auth_ids = a_authors
        b_auth_names, b_auth_ids = b_authors
        a_df = self.df.dropna(subset=[a_id, a_auth_names, a_auth_ids])
        b_df = a_df.dropna(subset=[b_id, b_auth_names, b_auth_ids])
    
        # verify the integrity of the author/id pairs
        a_mismatches = self._find_authors_mismatch(a_df, a_authors, a_id, delimiter)
        if len(a_mismatches) > 0 and self.verbose:
            warnings.warn(f'Found {len(a_mismatches)} dataset A paper(s) for which number of author names does not match number of author ids', RuntimeWarning)

        b_mismatches = self._find_authors_mismatch(b_df, b_authors, b_id, delimiter)
        if len(b_mismatches) > 0 and self.verbose:
            warnings.warn(f'Found {len(b_mismatches)} dataset B paper(s) for which number of author names does not match number of author ids', RuntimeWarning)
    
        # remove any papers that have bad author information
        b_df = b_df.loc[(~b_df[a_id].isin(set(a_mismatches))) & (~b_df[b_id].isin(set(b_mismatches)))].copy()
    
        # compute map of scopus paper ids to s2 paper ids
        matched_papers = {a_pid: b_pid for a_pid, b_pid in zip(b_df[a_id].to_list(), b_df[b_id].to_list())}

        # compute scopus author id to scopus author name
        aID_to_name = {}  # authID_to_name
        auth_list = [x.split(delimiter) for x in self.df[a_auth_names].to_list() if not pd.isna(x)]
        id_list = [x.split(delimiter) for x in self.df[a_auth_ids].to_list() if not pd.isna(x)]
        for id_sublist, auth_sublist in zip(id_list, auth_list):
            for auth_id, name in zip(id_sublist, auth_sublist):
                if auth_id not in aID_to_name:
                    aID_to_name[auth_id] = name
                    
        # scopus paper id to list of scopus authors ids (papers_authID_map)
        papers_aID_map = {k:list(set(v.split(delimiter))) for k,v in zip(b_df[a_id].to_list(), b_df[a_auth_ids].to_list())}
        
        # scopus author id to s2 paper id
        aID_to_s2 = {} 
        for k, v in matched_papers.items():
            auth_ids = papers_aID_map[k]
            for auth_id in auth_ids:
                if auth_id not in aID_to_s2:
                    aID_to_s2[auth_id] = {v}
                else:
                    aID_to_s2[auth_id].add(v)
        
        for k, v in aID_to_s2.items():  # convert sets into list for indexing
            aID_to_s2[k] = list(v)
        
        # compute map of s2 paper ids to s2 author ids
        b_papers_b_authID = {k:list(set(v.split(delimiter))) for k,v in zip(b_df[b_id].to_list(), b_df[b_auth_ids].to_list())}
        
        # compute s2 author id to s2 author name
        b_authID_name = {} 
        auth_list = [x.split(delimiter) for x in self.df[b_auth_names].to_list() if not pd.isna(x)]
        id_list = [x.split(delimiter) for x in self.df[b_auth_ids].to_list() if not pd.isna(x)]
        for id_sublist, auth_sublist in zip(id_list, auth_list):
            for auth_id, name in zip(id_sublist, auth_sublist):
                if auth_id not in b_authID_name:
                    b_authID_name[auth_id] = name

        # perform matching in parallel
        aid_list = list(aID_to_s2.keys())
        random.Random(42).shuffle(aid_list)  # add seed to random before shuffle
        kchunks = gen_chunks(aid_list, self.n_jobs)
        jobs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(match_authors)(kc, b_papers_b_authID, b_authID_name, aID_to_s2, aID_to_name, sub_table=self.CHAR_NORMALIZATION_DICT) for kc in kchunks)

        # join results and lookup remaining with text match
        errors = []
        auth_map = {}
        for am, e,_ in jobs:
            auth_map.update(am)
            errors += e
        
        if self.verbose:
            print(f'[Orca]: Found {len(auth_map)} author matches. . .')
        
        kchunks = gen_chunks(errors, self.n_jobs)
        jobs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(error_match)(kc, auth_map, b_papers_b_authID, b_authID_name, aID_to_s2, aID_to_name, sub_table=self.CHAR_NORMALIZATION_DICT) for kc in kchunks)
        
        match_df = {
            'SCOPUS_Author_ID': [],
            'SCOPUS_Author_Name': [],
            'S2_Author_ID': [],
            'S2_Author_Name': [],
        }

        auth_map = {k:v[0] for k,v in auth_map.items()}
        auth_map.update(known_matches)
        for k,v in auth_map.items():
            name = aID_to_name.get(v, 'Unknown')
            s2_name = b_authID_name.get(k, 'Unknown')

            match_df['SCOPUS_Author_ID'].append(v)
            match_df['SCOPUS_Author_Name'].append(name)
            match_df['S2_Author_ID'].append(k)
            match_df['S2_Author_Name'].append(s2_name)
        return pd.DataFrame.from_dict(match_df)

    
    def _find_authors_mismatch(self, df, auth_col, id_col, delimiter):
        mismatched = []
        auth_name, auth_id = auth_col
        ids = [[y for y in x.split(delimiter) if y not in ('None', 'Unknown')] for x in df[auth_id].to_list()]
        authors = [[y for y in x.split(delimiter) if y not in ('None', 'Unknown')] for x in df[auth_name].to_list()]
        for auth_ids, auth_names, paper_id in zip(ids, authors, df[id_col].to_list()):
            if len(auth_ids) != len(auth_names):
                mismatched.append(paper_id)
        return mismatched

        
    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs):
        cpu_count = multiprocessing.cpu_count()
        if not isinstance(n_jobs, int):
            raise ValueError(f'n_jobs must be an int')

        limit = cpu_count + n_jobs
        if (n_jobs == 0) or (limit < 0) or (2 * cpu_count < limit):
            raise ValueError(f'n_jobs must take a value on [-{cpu_count}, -1] or [1, {cpu_count}]')
        
        if n_jobs < 0:
            self._n_jobs = cpu_count - abs(n_jobs) + 1
        else:
            self._n_jobs = n_jobs