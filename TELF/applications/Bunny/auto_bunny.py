import os
import re
import sys
import pathlib
import pandas as pd
from dataclasses import dataclass, field

from .bunny import Bunny, BunnyFilter
from ..Cheetah import Cheetah
from ...pre_processing.iPenguin.Scopus import Scopus
from ...pre_processing.iPenguin.SemanticScholar import SemanticScholar
from ...pre_processing.Vulture import Vulture
from ...helpers.file_system import check_path_var as check_path

@dataclass
class AutoBunnyStep:
    """Class for keeping track of AutoBunny args"""
    modes: list
    max_papers: int = 2000
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
    
    def __init__(self, core, s2_key=None, scopus_keys=None, output_dir=None, cache_dir=None, cheetah_index=None, verbose=False, use_vulture_cheetah=True):
        self.core = core
        self.s2_key = s2_key
        self.scopus_keys = scopus_keys
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.cheetah_index = cheetah_index
        self.verbose = verbose
        self.use_vulture_cheetah = use_vulture_cheetah

        if self.verbose:
            print("[AutoBunny.__init__] Initialized")
            print(f"  s2_key={self.s2_key}")
            print(f"  scopus_keys={self.scopus_keys}")
            print(f"  output_dir={self.output_dir}")
            print(f"  cache_dir={self.cache_dir}")
            print(f"  cheetah_index={self.cheetah_index}")
            print(f"  core_rows={len(self.core)}")

    
    def run(self, 
            steps, 
            *, 
            s2_key=None, 
            scopus_keys=None, 
            cheetah_index=None, 
            max_papers=250000, 
            checkpoint=True,
            filter_type:str=None, # must be a key from Bunny.FILTERS
            filter_value=None):
        
        # validate input
        if not isinstance(steps, (list, tuple)):
            steps = [steps]
        for i,x in enumerate(steps):
            if not isinstance(x, AutoBunnyStep):
                raise ValueError(f'Step at index {i} in `steps` is not valid')
        if self.verbose:
            print(f"[AutoBunny.run] Steps to run: {len(steps)}")
            for idx, s in enumerate(steps):
                print(f"  Step {idx}: modes={s.modes}, max_papers={s.max_papers}, hop_priority={s.hop_priority}, "
                      f"cheetah_settings={s.cheetah_settings}, vulture_settings={s.vulture_settings}")
    
        if s2_key is not None:
            if self.verbose:
                print(f"[AutoBunny.run] Overriding s2_key")
            self.s2_key = s2_key
        if scopus_keys is not None:
            if self.verbose:
                print(f"[AutoBunny.run] Overriding scopus_keys")
            self.scopus_keys = scopus_keys
        if cheetah_index is not None:
            if self.verbose:
                print(f"[AutoBunny.run] Overriding cheetah_index")
            self.cheetah_index = cheetah_index
            
        # init search
        df = self.core
        cheetah_table = None
        
        # run for specified steps
        if self.verbose:
            print(f"[AutoBunny.run] Starting run loop with df size={len(df)}; checkpoint={checkpoint}; max_papers={max_papers}")

        print(len(steps), "steps to run")
        print(steps)
        for i, s in enumerate(steps):
            if self.verbose:
                print(f"\n[AutoBunny.run] ------- STEP {i} START -------")

            modes = s.modes
            if self.use_vulture_cheetah:
                cheetah_settings = s.cheetah_settings
                vulture_settings = s.vulture_settings
            step_max_papers = s.max_papers
            hop_priority = s.hop_priority
            hop = int(df.type.max())

            if self.verbose:
                print(f"[AutoBunny.run] Current hop={hop}")
                print(f"[AutoBunny.run] Modes={modes}")
                print(f"[AutoBunny.run] Step max_papers={step_max_papers}, hop_priority={hop_priority}")
                if self.use_vulture_cheetah:
                    print(f"[AutoBunny.run] cheetah_settings={cheetah_settings}")
                    print(f"[AutoBunny.run] vulture_settings={vulture_settings}")
            
            if checkpoint:
                path = os.path.join(self.output_dir, f'hop-{hop}.csv')
                if self.verbose:
                    print(f"[AutoBunny.run] Checkpoint: writing {path}")
                df.to_csv(path, index=False)

                if self.use_vulture_cheetah:
                    cheetah_settings['do_results_table'] = True
                    
                    if i == 0 and len(cheetah_settings) > 1:
                        if self.verbose:
                            print("[AutoBunny.run] Pre-step vulture + cheetah on initial df")
                        tmp_df = self.__vulture_clean(df, vulture_settings)
                        tmp_df, cheetah_table = self.__cheetah_filter(tmp_df, cheetah_settings)
                    if cheetah_table is not None:
                        path = os.path.join(self.output_dir, f'cheetah_table-{hop}.csv')
                        cheetah_table.to_csv(path, index=False)
                        if self.verbose:
                            print(f"[AutoBunny.run] Checkpoint: writing {path}")
            
            hop_estimate = Bunny.estimate_hop(df, modes[0]) # TODO: fix estimate_hop to use all modes
            if self.verbose:
                print(f"[AutoBunny.run] hop_estimate (next hop)={hop_estimate}")
            if hop_estimate > max_papers:
                msg = f'Early termination after {i} hops due to max papers in next hop'
                if self.verbose:
                    print(f"[AutoBunny.run] {msg} (max_papers={max_papers})", file=sys.stderr)
                print(msg, file=sys.stderr)
                return df
                
            if self.verbose:
                print("[AutoBunny.run] Executing bunny hop...")
            df = self.__bunny_hop(df, modes, step_max_papers, hop_priority)
            
            if filter_value and filter_type:
                if self.verbose:
                    print(f"[AutoBunny.run] Applying BunnyFilter {filter_type}={filter_value}")
                bunny = Bunny()
                query = BunnyFilter(filter_type, filter_value)
                subset_df = bunny.apply_filter(df, query, filter_in_core=True, do_author_match=False).reset_index(drop=True)
                if len(subset_df) < 1:
                    print("No papers for filter_value, using original df without filter.")
                    if self.verbose:
                        print("[AutoBunny.run] Filter produced empty set; keeping unfiltered df")
                else:
                    if self.verbose:
                        print(f"[AutoBunny.run] Filtered df size={len(subset_df)} (from {len(df)})")
                    df = subset_df

            if self.use_vulture_cheetah and self.verbose:
                print("[AutoBunny.run] Running vulture clean...")
            if self.use_vulture_cheetah:
                df = self.__vulture_clean(df, vulture_settings)

            if self.use_vulture_cheetah and self.verbose:
                print("[AutoBunny.run] Running cheetah filter...")
            if self.use_vulture_cheetah:
                df, cheetah_table = self.__cheetah_filter(df, cheetah_settings)
            
            # format df
            # if 'clean_title_abstract' in df.columns:
            #     if self.verbose:
            #         print("[AutoBunny.run] Dropping temporary column 'clean_title_abstract'")
            #     df.drop(columns=['clean_title_abstract'], inplace=True)
            df = df.reset_index(drop=True)

            if self.verbose:
                print(f"[AutoBunny.run] Step {i} complete; df size now {len(df)}")
                print(f"[AutoBunny.run] ------- STEP {i} END -------")
        
        # save final results if checkpointing
        if checkpoint:
            hop = int(df.type.max())
            final_papers_path = os.path.join(self.output_dir, 'final_bunny_papers.csv')
            if self.verbose:
                print(f"[AutoBunny.run] Final checkpoint: writing {final_papers_path}")
            df.to_csv(final_papers_path, index=False)

            if   self.use_vulture_cheetah and cheetah_table is not None:
                path = os.path.join(self.output_dir, f'cheetah_table-{hop}.csv')
                cheetah_table.to_csv(path, index=False)
                if self.verbose:
                    print(f"[AutoBunny.run] Final checkpoint: writing {path}")

                if self.verbose:
                    print("[AutoBunny.run] Building final cheetah table aggregation...")
                final_table = self.__final_cheetah_table()
                final_cheetah_path = os.path.join(self.output_dir, 'final_cheetah_table.csv')
                final_table.to_csv(final_cheetah_path, index=False)
                if self.verbose:
                    print(f"[AutoBunny.run] Final checkpoint: writing {final_cheetah_path}")

        print(len(df), "papers after all hops")
        if self.verbose:
            print("[AutoBunny.run] Completed all steps")
        return df
    
    
    ### Helpers
    
    
    def __final_cheetah_table(self, stem='cheetah_table'):
        if self.verbose:
            print(f"[__final_cheetah_table] Aggregating by stem='{stem}' in {self.output_dir}")
        files = [x for x in os.listdir(self.output_dir) if x.endswith('.csv') and stem in x]
        if self.verbose:
            print(f"[__final_cheetah_table] Found {len(files)} candidate files")

        frames = {}
        for f in files:
            match = re.search(f"{stem}-(\d+).csv", f)
            if match:
                x = int(match.group(1))
                path = os.path.join(self.output_dir, f)
                frames[x] = pd.read_csv(path)
                if self.verbose:
                    print(f"[__final_cheetah_table] Loaded hop={x} file '{f}' with {len(frames[x])} rows")

        for hop, df in frames.items():
            df = df[df.columns[:-2]].copy()
            num_papers_col = df.columns[-1]
            df.rename(columns={num_papers_col: f'hop{hop}-{num_papers_col}'}, inplace=True)
            frames[hop] = df
            if self.verbose:
                print(f"[__final_cheetah_table] Renamed count col for hop {hop} -> 'hop{hop}-{num_papers_col}'")

        frames = list(frames.values())
        df = frames[0]
        for tmp_df in frames[1:]:
            df = df.merge(tmp_df, on=list(df.columns[:2]), how='outer')
            if self.verbose:
                print(f"[__final_cheetah_table] Merged frame; current shape={df.shape}")
        return df
    
    
    def __bunny_hop(self, df, modes, max_papers, hop_priority):
        if self.verbose:
            print(f"[__bunny_hop] modes={modes}, max_papers={max_papers}, hop_priority={hop_priority}")
            print(f"[__bunny_hop] Using scopus={'yes' if self.scopus_keys is not None else 'no'}; cache_dir={self.cache_dir}")
        bunny = Bunny(s2_key=self.s2_key, output_dir=self.cache_dir, verbose=self.verbose)
        use_scopus = self.scopus_keys is not None
        hop_df = bunny.hop(df, 1, modes, use_scopus=use_scopus, filters=None, max_papers=max_papers, hop_priority=hop_priority,
                           scopus_keys=self.scopus_keys, s2_dir='s2', scopus_dir='scopus')
        if self.verbose:
            print(f"[__bunny_hop] Hop returned {len(hop_df)} rows")
        return hop_df
    
    
    def __cheetah_filter(self, df, cheetah_settings):
        if self.verbose:
            print(f"[__cheetah_filter] Settings: {cheetah_settings}")
            print(f"[__cheetah_filter] Input df size={len(df)}")
    
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

        if self.verbose:
            print(f"[__cheetah_filter] df_prev={len(df_prev)}, df_curr={len(df_curr)}, max_type={max_type}")
    
        # setup cheetah 
        cheetah = Cheetah(verbose=self.verbose)
        index_file = os.path.join(self.output_dir, 'cheetah_index.p')
        if self.verbose:
            print(f"[__cheetah_filter] Indexing df_curr to {index_file} with columns={cheetah_columns}")
        cheetah.index(df_curr, 
                      columns=cheetah_columns, 
                      index_file=index_file,
                      reindex=True)
        
        # filter with cheetah
        if self.verbose:
            print(f"[__cheetah_filter] Searching with cheetah_settings={cheetah_settings}")
        cheetah_df, cheetah_table = cheetah.search(**cheetah_settings)
        if self.verbose:
            print(f"[__cheetah_filter] cheetah_df size={len(cheetah_df)}")
            if cheetah_table is not None:
                print(f"[__cheetah_filter] cheetah_table size={len(cheetah_table)}")
        
        # fix the cheetah_table (if being computed)
        if cheetah_table is not None and not cheetah_table.empty:
            if self.verbose:
                print("[__cheetah_filter] Rewriting indices to s2ids in cheetah_table")
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
            if self.verbose:
                print("[__cheetah_filter] cheetah_table indices updated")
        
        # combine cheetah filter results with frozen results from previous hops
        cheetah_df = pd.concat([df_prev, cheetah_df], ignore_index=True)
        cheetah_df = cheetah_df.drop_duplicates(subset=['s2id'], keep='first')
        cheetah_df = cheetah_df.reset_index(drop=True)
        if self.verbose:
            print(f"[__cheetah_filter] Combined cheetah_df size={len(cheetah_df)} (after concat + dedupe)")
        return cheetah_df, cheetah_table
    
    
    def __vulture_clean(self, df, vulture_settings):
        if self.verbose:
            print(f"[__vulture_clean] Input size={len(df)}; settings={vulture_settings}")
        
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

        cleaned = vulture.clean_dataframe(**dataframe_clean_args)
        if self.verbose:
            print(f"[__vulture_clean] Output size={len(cleaned)}")
        return cleaned
    
    
    
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
                    raise ValueError(f'The key "{key}" was rejected by the Scopus API')
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

    def __process_path(self, path, var_name):
        if path is None:
            return pathlib.Path('/tmp')
        elif isinstance(path, str):
            _path = pathlib.Path(path)
        elif isinstance(path, pathlib.Path):
            _path = path
        else:
            raise TypeError(f'Unsupported type "{type(path)}" for `{var_name}`')
        check_path(_path, var_name)
        return _path
            
    @output_dir.setter
    def output_dir(self, output_dir):
        self._output_dir = self.__process_path(output_dir, 'output_dir')
        
    @cache_dir.setter
    def cache_dir(self, cache_dir):
        self._cache_dir = self.__process_path(cache_dir, 'cache_dir')
