import os
import sys
import json
import time
import httpx
import asyncio
import pathlib
import warnings
import pandas as pd
from tqdm import tqdm
import multiprocessing
from collections import Counter
from joblib import Parallel, delayed

# import interal libraries
from .scopusAPI import ScopusAPI
from .parser import ScopusQueryParser
from .process_scopus_json import form_df
from ....helpers.data_structures import gen_chunks

class Scopus:
    
    MODES = {'fs'}
    
    # Scopus sets a sneaky search limit on how many papers can be acquired
    # This search limit can be overcome (in some cases) but requires some extra work
    SEARCH_LIMIT = 2000
    QUERY_LIMIT = 10000  # how many different parameters can be in a single query
    
    # Set a verbosity level that will trigger debug print
    DEBUG_MODE = 10
    
    def __init__(self, 
                 keys: list,
                 mode: str = 'fs', 
                 name: str = 'scopus', 
                 n_jobs: int = -1,
                 ignore = None,
                 verbose: bool = False):
        """
        Initializes the iPenguin.Scopus instance with the specified keys, mode, and optional settings.

        This constructor method sets up the Scopus instance by initializing the keys, mode, name, papers to ignore,
        verbosity of output, and number of jobs for parallel processing (if applicable). It then attempts to 
        establish a connection to the Scopus API to validate the keys

        Parameters:
        -----------
        keys: list
            The list of API keys to be used for this instance. The list must contain one or more valid API keys.
            An error will be thrown if any of the keys fail to validate.
        mode: str, (optional)
            The mode in which the Scopus instance should operate in. Currently one mode is supported, 'fs'. This
            is the file system mode and will download papers to the directory path provided with `name`.
        name: str, (optional):
            The name associated with the instance. In the case of 'fs' `mode` (the default), this parameter is 
            expected to be the path to the directory where the downloaded files will be saved
        ignore: set, list, Iterable, None, (optional)
            This parameter allows for certain papers to be skipped and not downloaded. This is useful for speeding
            up download times by skipping previously downloaded papers and saving on API keys. If None, the 
            ignore papers will be determined depending on the mode that this instance is operating in. If defined,
            this parameter needs to be a data structure that has the __contains__ method implemented. Default is
            None.
        n_jobs: int, (optional)
            The number of jobs for parallel processing. Default is -1 (use all available cores).
        verbose: bool, int (optional)
            If set to True, the class will print additional output for debugging or information purposes. 
            Can also be an int where verbose >= 1 means True with a higher integer controlling the level 
            of verbosity. Values above 10 will activate debug for this higher level class. Values 
            above 100 will activate debug print for the lower level ScopusAPI class. Values above 1000 
            will provide a full debug print and should only be used for testing. Default is True.
        """
        self.key = None
        self.api = None
        self.keys = keys
        self.mode = mode
        self.name = name
        self.n_jobs = n_jobs
        self.ignore = ignore
        self.pbar_count = 0
        self.verbose = verbose

        # create dictionaries of helper functions for supported modes
        prefixes = ['_prepare_', '_save_', '_read_']
        for p in prefixes:
            setattr(self, f"{p[1:-1]}_funcs", self.__generate_func_dict(p))

            
    @classmethod
    def get_df(cls, path, targets=None, *, backend='threading', n_jobs=1, verbose=False):
        """
        Parallelized function for generating a pandas DataFrame from Scopus JSON data. The data  
        is also processed to adhere to SLIC standards. 

        Parameters
        ----------
        path : str, pathlib.Path
            The path to the directory where the JSON files are stored
        targets : set, list; optional
            A collection of eids (Scopus paper unique identifiers) that need to be present in the 
            DataFrame. A warning will be provided if all targets cannot be found at the given path.
            If targets is None, no filtering is performed and the DataFrame is formed from all json
            data stored at `path`.
        n_jobs : int; optional
            How many parallel processes to use for this function. Default is 1 (single core)

        Returns
        -------
        pd.DataFrame
            Resulting DataFrame. If no valid files are able to processed to form the DataFrame, 
            this function will return None
        targets
            The list of Scopus paper ids that were expected to be downloaded
        """
        path = pathlib.Path(path)
        if not path.is_dir():
            raise ValueError('The `path` parameter must be a path to an existing directory')
        if not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError('`n_jobs` must be a positive integer')
        
        if targets is not None:
            files = [file for name in targets for file in path.glob(f"{name}.json")]
            if len(files) != len(targets):
                warnings.warn(f'[Scopus]: Requested {len(targets)} papers but only found ' \
                              f'{len(files)} stored at `path`')
        else:
            files = list(path.glob("*.json"))

        if not files:  # if no valid files found, terminate execution here by returning None
            return None, targets

        # chunk the files for parallel processing
        n_jobs = min(n_jobs, len(files))  
        chunks = gen_chunks(files, n_jobs)
        
        # generate dataframe slices in parallel
        jobs = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(delayed(form_df)(c) for c in chunks)
        df = pd.concat(jobs, sort=False)
        df = df.reset_index(drop=True)
        return df, targets
    
    
    def split_query(self, expression: str) -> list:
        def helper(query):
            parser = ScopusQueryParser()
            query_str = parser.to_string(query)
            
            case1 = len(query) < self.QUERY_LIMIT  # base case #1
            if case1:
                
                ### temporary workaround for api count failing with large queries
                # when a query has a large number of elements that each produce multiple papers (lets say many authors)
                # being joined by OR, Scopus cannot process the query and simply times out during the attempt. This
                # heuristic will try to avoid this by splitting the query if it is too long and recieving an erronous
                # count from Scopus (-1) 
                if len(query) > 500:  # arbitrary big number
                    if isinstance(self.verbose, int) and self.verbose >= self.DEBUG_MODE:
                        print(f'[DEBUG]: Query too long! api_count manually set to SEARCH LIMIT!', file=sys.stderr)
                    api_count = self.SEARCH_LIMIT
                else:
                    api_count = self.count(query_str, False)
                    if api_count == -1:  # temporary solution for api not returning count
                        if isinstance(self.verbose, int) and self.verbose >= self.DEBUG_MODE:
                            print(f'[DEBUG]: API returned -1! api_count manually set to SEARCH LIMIT!', file=sys.stderr)
                        api_count = self.SEARCH_LIMIT
                
                #case2 = api_count < self.SEARCH_LIMIT  # base case #2
                case2 = True  # temporarily dropping second condition to test cursor pagination
                if isinstance(self.verbose, int) and self.verbose >= self.DEBUG_MODE:
                    print(f'[DEBUG]: split_query() - {len(query)=}, {api_count=}', file=sys.stderr)
                if case2:
                    if isinstance(self.verbose, int) and self.verbose >= self.DEBUG_MODE:
                        print(f'[DEBUG]: Successfully reached base case!', file=sys.stderr)
                    return [query]

            split = parser.split(query)
            s1, s2 = split[0], split[1]
            if s2 is None:  # could not split previous query, terminate
                if isinstance(self.verbose, int) and self.verbose >= self.DEBUG_MODE:
                    print(f'[DEBUG]: Cannot further split the query {parser.to_string(s1)!r}! Skipping!', file=sys.stderr)
                return [s1]
            else:
                return helper(s1) + helper(s2)

        # process the input string and start the recursion
        parser = ScopusQueryParser()
        query = parser.parse(expression)
        split = helper(query)
        return [parser.to_string(x) for x in split]
    
    
    def count(self, query, verbose=None):
        self.__pop_key()
        
        if verbose is None:
            verbose = self.verbose
        if verbose:
            start = time.perf_counter()
        
        num_papers =  asyncio.run(self.num_query_coroutine(query), debug=False)
                
        if verbose:
            end = time.perf_counter()
            print(f"[Scopus]: Found {num_papers:,} papers in {end - start:.2f}s", file=sys.stderr)
        return num_papers
    
    
    async def num_query_coroutine(self, query):
        async with httpx.AsyncClient() as client:
            api = ScopusAPI(client, key=self.key)
            await api.get_num_papers_from_query(query)
            count = await asyncio.gather(api.run(), self.__num_papers_coroutine(api.results)) # start both handlers concurrently
            count = count[1]  # api.run() does not return anything, get the actual results at index 1
        return int(count)
    
    
    def search(self, query, *, n=100):
        self.__pop_key()
        if self.verbose:
            start = time.perf_counter()
            self.pbar_count = 0

        # split queries to avoid Scopus search limits
        parser = ScopusQueryParser()
        queries = self.split_query(query)
        paper_counts = [self.count(q, False) for q in queries]
        for c, q in zip(paper_counts, queries):
            if isinstance(self.verbose, int) and self.verbose >= self.DEBUG_MODE:
                print(f'[DEBUG]: search() - query= \n\n{q}\n', file=sys.stderr)
            
            # get the parsed version the string for length check
            q_parsed = parser.parse(q)
            
            if c < 0:
                raise ValueError(f'Scopus could not accept the subquery {q[:150] + "... (truncated)" if len(q) > 150 else q!r}. ' \
                                 f'This means that this query is either too long or poorly formatted')
            elif len(q_parsed) > self.QUERY_LIMIT:
                raise ValueError(f'One or more subqueries exceed the Scopus query limit of ' \
                                 f'{self.SEARCH_LIMIT} parameters and cannot be chunked any further')
            elif c > self.SEARCH_LIMIT and (n == 0 or n > self.SEARCH_LIMIT):
                warnings.warn(f'[Scopus]: The subquery {q[:150] + "... (truncated)" if len(q) > 150 else q!r} ' \
                              f'exceeds the Scopus search limit. Only the top {self.SEARCH_LIMIT} papers will be returned for this subquery.')
        
        num_papers = sum(paper_counts)
        if num_papers == 0:
            raise ValueError('No Scopus papers for `query` can be found. If you expect ' \
                             'results for this query, check your syntax')
        
        self.paper_ids = set()
        self.query_index = 0
        while True:

            # prepare search 
            self.__pop_key()
            ignore = self.__prepare()
            
            # search
            try:
                asyncio.run(self.search_coroutine(queries[self.query_index:], 
                                                  ignore, n, num_papers), debug=False)
            except KeyboardInterrupt:
                asyncio.run(self.cleanup_coroutine(), debug=False)
                raise KeyboardInterrupt

            # if verbose, wait for all scopusAPI messages to finish printing
            if self.verbose:
                time.sleep(1)
                
            # process
            if ScopusAPI.get_quota(self.key) <= 0:
                warnings.warn(f'[Scopus]: Hit Scopus API quota with {self.key!r}') 
                self.key = None # hit key quota
            elif len(self.paper_ids) < num_papers:
                warnings.warn('[Scopus]: Scopus API returned with less papers than expected. . .') 
                break
            else:
                break


        if self.verbose:
            end = time.perf_counter()
            print(f"[Scopus]: Finished downloading {len(self.paper_ids):,} "
                  f"papers in {end - start:.2f}s",  file=sys.stderr)

        df = self.__read(list(self.paper_ids))
        return df
    
    
    async def search_coroutine(self, data, ignore, n, num_papers):
        if self.api is not None:
            await self.cleanup_coroutine()
        
        client = httpx.AsyncClient()
        self.api = ScopusAPI(client, key=self.key, ignore=ignore, verbose=self.verbose)

        for i, d in enumerate(data):
            if isinstance(self.verbose, int) and self.verbose >= self.DEBUG_MODE:
                print(f'[DEBUG]: search_coroutine() - query = {d}\n', file=sys.stderr)
            await self.api.find_papers_by_query(d, n=n)#n=min(n, self.SEARCH_LIMIT))

            #  start both handlers concurrently
            results = await asyncio.gather(self.api.run(), self.__search_coroutine(self.api.results, num_papers)) 
            papers, errno = results[1]  # api.run() does not return anything, get the actual results at index 1    
            
            self.paper_ids |= papers
            if errno != 0:
                if self.verbose:
                    print(f'[Scopus]: Early search termination with errno {errno}', file=sys.stderr)
                break
            else:  # no error, update query index
                self.query_index += 1
    
        await self.cleanup_coroutine()
    
    
    async def cleanup_coroutine(self):
        if self.api is not None:
            await self.api.cleanup()
            self.api = None
    
    
    ### Helpers
    
    
    def __pop_key(self):
        if self.key is None:
            try:
                self.key = self.keys.pop()
            except IndexError:
                self.key = None
                
        if self.key is None:
            raise KeyError('No more Scopus API keys left to try')
        
            
    def __generate_func_dict(self, prefix):
        return {
            self.__strip_prefix(name, prefix): getattr(self, name) for name in dir(self)
            if self.__is_valid_function(name, prefix)
        }
    
    def __strip_prefix(self, name, prefix):
        return name[len(prefix):]

    
    def __is_valid_function(self, name, prefix):
        return (name.startswith(prefix) and callable(getattr(self, name))) and \
                self.__strip_prefix(name, prefix) in Scopus.MODES
                
    
    async def __search_coroutine(self, queue, num_papers):
        use_pbar = bool(self.verbose)
        if use_pbar:
             pbar = tqdm(initial=len(self.paper_ids), total=num_papers)
        
        errno = 0
        found_papers = []
        while True:
            try:
                result = await queue.get()
                if isinstance(result, int):
                    errno = result
                    if use_pbar:
                        pbar.close()
                    break
                
                op, pid, paper = result
                if use_pbar and pid not in self.paper_ids:
                    self.pbar_count += 1
                    pbar.update(1)
                
                found_papers.append(pid)
                if paper is not None:
                    self.__save(result)
                
                queue.task_done()  # signal that the queue item has been processed
            except asyncio.CancelledError:
                return set(found_papers), errno
        
        if isinstance(self.verbose, int) and self.verbose >= self.DEBUG_MODE:
            pid_counts = Counter(found_papers)
            duplicate_count = sum(1 for count in pid_counts.values() if count > 1)
            print(f'[DEBUG]: __search_coroutine() - ScopusAPI has returned {duplicate_count} duplicate papers', file=sys.stderr)
            
            # determine the maximum length of the item names for alignment
            max_pid_length = max(len(pid) for pid in pid_counts)

            # pretty print debug info
            for pid, count in pid_counts.items():
                if count > 1:
                    print(f'{pid:<{max_pid_length}} - {count:>4}', file=sys.stderr)
                    
        return set(found_papers), errno
    
    
    async def __num_papers_coroutine(self, queue):
        count = -1
        while True:
            try:
                result = await queue.get()
                if isinstance(result, int) and result == 0:
                    break
                
                _, _, count = result
                queue.task_done()  # signal that the queue item has been processed
            except asyncio.CancelledError:
                return count
        return count
            
        
    def __prepare(self):
        prepare_func = self.prepare_funcs[self.mode]
        return prepare_func()
    
    
    def _prepare_fs(self):
        path = pathlib.Path(self.name)
        if not path.exists(): 
            os.makedirs(path)
        elif path.exists() and not path.is_dir():
            raise ValueError(f'The path {path!r} is not a directory')
        
        if self.ignore is not None:
            return self.ignore
        else:
            files = {f.stem for f in path.glob('*.json')}
            return {x[7:] for x in files}
    
    
    def __save(self, data):
        save_func = self.save_funcs[self.mode]
        return save_func(data)
    
    
    def _save_fs(self, data):
        op, eid, paper = data
        with open(os.path.join(self.name, f'{eid}.json'), 'w') as fh:
            json.dump(paper, fh, indent=4)
         
        
    def __read(self, paper_ids):
        read_func = self.read_funcs[self.mode]
        return read_func(paper_ids)
    
    
    def _read_fs(self, paper_ids):
        return self.get_df(self.name, targets=paper_ids, n_jobs=self.n_jobs, verbose=self.verbose)

    
    ### Setters / Getters


    @property
    def n_jobs(self):
        return self._n_jobs
    
    @property
    def keys(self):
        return self._keys
    
    
    @keys.setter
    def keys(self, keys):
        for k in keys:
            try:
                with httpx.Client() as client:
                    api = ScopusAPI(client, key=k)
            except ValueError:
                raise ValueError(f'The key "{k}" was rejected by the Scopus API')
        self._keys = keys.copy()
    
    
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