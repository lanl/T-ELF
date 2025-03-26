import os
import re
import sys
import pathlib
import pandas as pd
from dataclasses import dataclass, field

from .bunny import Bunny
from ..Cheetah import Cheetah
from ...pre_processing.iPenguin.Scopus import Scopus
from ...pre_processing.iPenguin.SemanticScholar import SemanticScholar
from ...pre_processing.Vulture import Vulture

@dataclass
class AutoBunnyStep:
    """Class for keeping track of AutoBunny args"""
    modes: list
    max_papers: int = 0
    hop_priority: str = 'random'
    cheetah_settings: dict = field(default_factory = lambda: {'query': None})
    vulture_settings: list = field(default_factory = lambda: [])

    
class AutoBunny:
    
    CHEETAH_INDEX = {
        'title': None, 
        'abstract': 'clean_title_abstract',
        'year': 'year',
        'author_ids': 'author_ids',
        'affiliations': 'affiliations',
        'country': 'affiliations',
    }
    
    def __init__(self, core, s2_key=None, scopus_keys=None, output_dir=None, cache_dir=None, cheetah_index=None, verbose=False):
        self.core = core
        self.s2_key = s2_key
        self.scopus_keys = scopus_keys
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.cheetah_index = cheetah_index
        self.verbose = verbose
        
    
    def run(self, steps, *, s2_key=None, scopus_keys=None, cheetah_index=None, max_papers=250000, checkpoint=True):
        
        # validate input
        if not isinstance(steps, (list, tuple)):
            steps = [steps]
        for i,x in enumerate(steps):
            if not isinstance(x, AutoBunnyStep):
                raise ValueError(f'Step at index {i} in `steps` is not valid')
    
        if s2_key is not None:
            self.s2_key = s2_key
        if scopus_keys is not None:
            self.scopus_keys = scopus_keys
        if cheetah_index is not None:
            self.cheetah_index = cheetah_index
            
        # init search
        df = self.core
        cheetah_table = None
        
        # run for specified steps
        for i, s in enumerate(steps):            
            modes = s.modes
            cheetah_settings = s.cheetah_settings
            vulture_settings = s.vulture_settings
            step_max_papers = s.max_papers
            hop_priority = s.hop_priority
            hop = int(df.type.max())
            
            if checkpoint:
                df.to_csv(os.path.join(self.output_dir, f'hop-{hop}.csv'), index=False)
                cheetah_settings['do_results_table'] = True
                
                if i == 0 and len(cheetah_settings) > 1:
                    tmp_df = self.__vulture_clean(df, vulture_settings)
                    tmp_df, cheetah_table = self.__cheetah_filter(tmp_df, cheetah_settings)
                if cheetah_table is not None:
                    cheetah_table.to_csv(os.path.join(self.output_dir, f'cheetah_table-{hop}.csv'), index=False)
            
            hop_estimate = Bunny.estimate_hop(df, modes[0]) # TODO: fix estimate_hop to use all modes
            if hop_estimate > max_papers:
                print(f'Early termination after {i} hops due to max papers in next hop', file=sys.stderr)
                return df
                
            df = self.__bunny_hop(df, modes, step_max_papers, hop_priority)
            df = self.__vulture_clean(df, vulture_settings)
            df, cheetah_table = self.__cheetah_filter(df, cheetah_settings)
            
            # format df
            df.drop(columns=['clean_title_abstract'], inplace=True)
            df = df.reset_index(drop=True)
        
        # save final results if checkpointing
        if checkpoint:
            hop = int(df.type.max())
            df.to_csv(os.path.join(self.output_dir, 'final_bunny_papers.csv'), index=False)
            if cheetah_table is not None:
                cheetah_table.to_csv(os.path.join(self.output_dir, f'cheetah_table-{hop}.csv'), index=False) 
                final_table = self.__final_cheetah_table()
                final_table.to_csv(os.path.join(self.output_dir, 'final_cheetah_table.csv'), index=False) 
        return df
    
    
    ### Helpers
    
    
    def __final_cheetah_table(self, stem='cheetah_table'):
        files = [x for x in os.listdir(self.output_dir) if x.endswith('.csv') and stem in x]
        frames = {}
        for f in files:
            match = re.search(f"{stem}-(\d+).csv", f)
            if match:
                x = int(match.group(1))
                frames[x] = pd.read_csv(os.path.join(self.output_dir, f))

        for hop, df in frames.items():
            df = df[df.columns[:-2]].copy()
            num_papers_col = df.columns[-1]
            df.rename(columns={num_papers_col: f'hop{hop}-{num_papers_col}'}, inplace=True)
            frames[hop] = df

        frames = list(frames.values())
        df = frames[0]
        for tmp_df in frames[1:]:
            df = df.merge(tmp_df, on=list(df.columns[:2]), how='outer')
        return df
    
    
    def __bunny_hop(self, df, modes, max_papers, hop_priority):
        bunny = Bunny(s2_key=self.s2_key, output_dir=self.cache_dir, verbose=self.verbose)
        use_scopus = self.scopus_keys is not None
        hop_df = bunny.hop(df, 1, modes, use_scopus=use_scopus, filters=None, max_papers=max_papers, hop_priority=hop_priority,
                           scopus_keys=self.scopus_keys, s2_dir='s2', scopus_dir='scopus')
        return hop_df
    
    
    def __cheetah_filter(self, df, cheetah_settings):
    
        # index settings 
        cheetah_columns = {
            'title': None, 
            'abstract': 'clean_title_abstract',
            'year': 'year',
            'author_ids': 'author_ids',
            'affiliations': 'affiliations',
            'country': 'affiliations',
        }
    
        # preserve the previously filtered papers
        max_type = df.type.max()
        df_prev = df.loc[df.type < max_type]
        df_curr = df.loc[df.type == max_type]
    
        # setup cheetah 
        cheetah = Cheetah(verbose=self.verbose)
        index_file = os.path.join(self.output_dir, 'cheetah_index.p')
        cheetah.index(df_curr, 
                      columns=cheetah_columns, 
                      index_file=index_file,
                      reindex=True)
        
        # filter with cheetah
        cheetah_df, cheetah_table = cheetah.search(**cheetah_settings)
        
        # fix the cheetah_table (if being computed)
        # the cheetah table uses indices set by df. These indices will be reset by the rest of
        # this function. It is more robust to replace indices with s2ids.
        if cheetah_table is not None and not cheetah_table.empty:
            cheetah_table['included_ids'] = cheetah_table.included_ids.fillna('').str.split(';')\
                .apply(lambda x: [int(i) for i in x if i] if x else [])

            def include_s2ids(indices):
                if not indices:
                    return None
                return ';'.join(map(str, df_curr.loc[indices].s2id.to_list()))
            
            def exclude_s2ids(indices):
                all_s2ids = {x for x in df_curr.s2id.to_list() if not pd.isna(x)}
                if not indices:
                    return ';'.join(list(all_s2ids))
                curr_s2ids = set(df_curr.loc[indices].s2id.to_list())
                return ';'.join(list(all_s2ids - curr_s2ids)) or None
            
            cheetah_table['selected_s2ids'] = cheetah_table.included_ids.apply(include_s2ids)
            cheetah_table['excluded_s2ids'] = cheetah_table.included_ids.apply(exclude_s2ids)
            cheetah_table = cheetah_table.drop(columns='included_ids')
        
        # combine cheetah filter results with frozen results from previous hops
        cheetah_df = pd.concat([df_prev, cheetah_df], ignore_index=True)
        cheetah_df = cheetah_df.drop_duplicates(subset=['s2id'], keep='first')
        cheetah_df = cheetah_df.reset_index(drop=True)
        return cheetah_df, cheetah_table
    
    
    def __vulture_clean(self, df, vulture_settings):
        
        # setup vulture
        vulture = Vulture(n_jobs=-1, cache=self.output_dir, verbose=self.verbose)
        
        dataframe_clean_args = {
            "df": df,
            "columns": ['title', 'abstract'],
            "append_to_original_df": True,
            "concat_cleaned_cols": True,
        }
        if vulture_settings:
            dataframe_clean_args["steps"] = vulture_settings
        return vulture.clean_dataframe(**dataframe_clean_args)
    
    
    
    ### Getters / Setters
    
    
    @property
    def core(self):
        return self._core
    
    @property
    def s2_key(self):
        return self._s2_key

    @property
    def scopus_keys(self):
        return self._scopus_keys

    @property
    def cheetah_index(self):
        return self._cheetah_index
    
    @property
    def output_dir(self):
        return self._output_dir
    
    @property
    def cache_dir(self):
        return self._cache_dir
    
    @core.setter
    def core(self, core):
        if not isinstance(core, pd.DataFrame):
            raise ValueError('AutoBunny expects core to be a SLIC DataFrame!')
        if 'type' not in core:
            core['type'] = [0] * len(core)
        self._core = core
    
    @s2_key.setter
    def s2_key(self, key):
        if key is not None:
            self._s2_key = key
        elif isinstance(key, str):
            try:
                ip = SemanticScholar(key=key)
                self._s2_key = key
            except ValueError:
                raise ValueError(f'The key "{key}" was rejected by the Semantic Scholar API')
        else:
            raise TypeError(f'Unsupported type "{type(key)}" for Semantic Scholar key')
        
    @scopus_keys.setter
    def scopus_keys(self, scopus_keys):
        if scopus_keys is None:
            self._scopus_keys = scopus_keys
        elif isinstance(scopus_keys, (list, set)):
            for key in scopus_keys:
                try:
                    ip = Scopus(keys=[key])
                except ValueError:
                    raise ValueError(f'The key "{k}" was rejected by the Scopus API')
            self._scopus_keys = list(scopus_keys)
        else:
            raise TypeError(f'Unsupported type "{type(key)}" for Scopus key')
            
    @cheetah_index.setter
    def cheetah_index(self, cheetah_index):
        if cheetah_index is None:
            self._cheetah_index = self.CHEETAH_INDEX
        elif isinstance(cheetah_index, dict):
            if not all(key in self.CHEETAH_INDEX for key in cheetah_index.keys()):
                raise ValueError(f'Invalid index key in `cheetah_index`. Valid keys are in '
                                 f'{list(self.CHEETAH_INDEX.keys())}')
                
            # fill in any missing keys from cheetah_index with default
            self._cheetah_index = {**self.CHEETAH_INDEX, **cheetah_index} 
        else:
            raise TypeError(f'Unsupported type "{type(cheetah_index)}" for `cheetah_index`')
            
    def __check_path(self, path, var_name):      
        if path.exists() and path.is_file():  # handle the path already existing as file
            raise ValueError(f'The path `{var_name}` points to a file instead of a directory')
        if not path.exists():
            path.mkdir(parents=True)  # parents=True ensures all missing parent directories are also created

    def __check_path(self, path, var_name):
        """
        Checks and ensures the given path exists as a directory. If path does not exist, a new directory
        will be created. If the path exists but is a file, a ValueError will be raised. A TypeError is
        raised if the provided path is neither a string nor a `pathlib.Path` object.
    
        Parameters:
        -----------
        path: str, pathlib.Path
            The path to be checked and ensured as a directory.
        
        Raises:
        -------
        TypeError:
            If the provided path is neither a string nor a `pathlib.Path` object.
        ValueError: 
            If the path points to an existing file.
        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not isinstance(path, pathlib.Path):
            raise TypeError(f'Unsupported type "{type(path)}" for `path`')
        path = path.resolve()
        if path.exists():
            if path.is_file():
                raise ValueError(f'`{var_name}` points to a file instead of a directory')
        else:
            path.mkdir(parents=True, exist_ok=True)

    def __process_path(self, path, var_name):
        if path is None:
            return pathlib.Path('/tmp')
        elif isinstance(path, str):
            _path = pathlib.Path(path)
        elif isinstance(path, pathlib.Path):
            _path = path
        else:
            raise TypeError(f'Unsupported type "{type(path)}" for `{var_name}`')
        self.__check_path(_path, var_name)
        return _path
            
    @output_dir.setter
    def output_dir(self, output_dir):
        self._output_dir = self.__process_path(output_dir, 'output_dir')
        
    @cache_dir.setter
    def cache_dir(self, cache_dir):
        self._cache_dir = self.__process_path(cache_dir, 'cache_dir')
        
