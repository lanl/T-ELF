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
from joblib import Parallel, delayed

# import interal libraries
from .ostiAPI import OSTIApi
from ..utils import gen_chunks
from .process_osti_json import form_df

class OSTI:
    
    MODES = {'fs'}
    
    def __init__(self, key: str | None = None, mode: str = 'fs', name: str = 'osti', n_jobs: int = -1, verbose: bool = False):
        self.key = key
        self.mode = mode
        self.name = name
        self.n_jobs = n_jobs
        self.verbose = verbose

        # create dictionaries of helper functions for supported modes
        prefixes = ['_prepare_', '_save_', '_read_']
        for p in prefixes:
            setattr(self, f"{p[1:-1]}_funcs", self.__generate_func_dict(p))
            
            
    @classmethod
    def get_df(cls, path, targets=None, *, backend='threading', n_jobs=1, verbose=False):
        """
        Parallelized function for generating a pandas DataFrame from OSTI JSON data. 

        Parameters
        ----------
        path : str, pathlib.Path
            The path to the directory where the JSON files are stored
        targets : set, list; optional
            A collection of OSTI_ids that need to be present in the DataFrame. A warning will be 
            provided if all targets cannot be found at the given path. If targets is None, 
            no filtering is performed and the DataFrame is formed from all json data stored at `path`.
        n_jobs : int; optional
            How many parallel processes to use for this function. Default is 1 (single core)

        Returns
        -------
        pd.DataFrame
            Resulting DataFrame. If no valid files are able to processed to form the DataFrame, 
            this function will return None
        """
        path = pathlib.Path(path)
        if not path.is_dir():
            raise ValueError('The `path` parameter must be a path to an existing directory')
        if not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError('`n_jobs` must be a positive integer')
        
        if targets is not None:
            files = [file for name in targets for file in path.glob(f"{name}.json")]
            if len(files) != len(targets):
                warnings.warn(f'[OSTI]: Requested {len(targets)} papers but only found ' \
                              f'{len(files)} stored at `path`')
        else:
            files = list(path.glob("*.json"))

        if not files:  # if no valid files found, terminate execution here by returning None
            return None

        # chunk the files for parallel processing
        n_jobs = min(n_jobs, len(files))  
        chunks = gen_chunks(files, n_jobs)
        
        # generate dataframe slices in parallel
        jobs = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(delayed(form_df)(c) for c in chunks)
        df = pd.concat(jobs, sort=False)
        df = df.reset_index(drop=True)
        return df
    
    
    def count(self, data, mode, verbose=None):
        if verbose is None:
            verbose = self.verbose
        if verbose:
            start = time.perf_counter()
        
        num_papers =  asyncio.run(self.count_coroutine(data, mode), debug=False)
                
        if verbose:
            end = time.perf_counter()
            print(f"[OSTI]: Found {num_papers:,} papers in {end - start:.2f}s", file=sys.stderr)
        return num_papers
    
    
    async def count_coroutine(self, data, mode):
        async with httpx.AsyncClient() as client:
            api = OSTIApi(client)
            
            if mode == 'paper':
                return len(data)
            elif mode == 'query':
                await api.find_papers_by_query(data, number=True)
            else:
                raise ValueError('Invalid OSTI mode selected')
            count = await asyncio.gather(api.run(), self.__count_coroutine(api.results)) # start both handlers concurrently
            count = count[1]  # api.run() does not return anything, get the actual results at index 1
        return int(count)
    
    
    def search(self, data, mode, *, n=100):
        if self.verbose:
            start = time.perf_counter()
            self.pbar_count = 0
        
        num_papers = self.count(data, mode, verbose=False)
        if num_papers == 0:
            raise ValueError('No OSTI papers for `data` can be found. If you expect results for this search, check your syntax')
        elif mode == 'query' and (n < num_papers):
            warnings.warn(f'[OSTI]: Found {num_papers} papers for query but limit is set at {n}', RuntimeWarning)
            num_papers = n
        
        # prepare search
        ignore = self.__prepare()
        
        # search
        paper_ids = asyncio.run(self.search_coroutine(data, mode, ignore, n, num_papers), debug=False)
            
        if self.verbose:
            end = time.perf_counter()
            time.sleep(1)  # if verbose, wait for all OSTIApi messages to finish printing
            print(f"[OSTI]: Finished downloading {len(paper_ids):,} papers in {end - start:.2f}s",  file=sys.stderr)
    
        # process
        df = self.__read(paper_ids)
        return df
    
    
    async def search_coroutine(self, data, mode, ignore, n, num_papers):
        async with httpx.AsyncClient() as client:
            api = OSTIApi(client, ignore=ignore)
            
            if mode == 'paper':
                await api.find_papers_by_id(data)
            elif mode == 'query':
                await api.find_papers_by_query(data, n=n)
            else:
                raise ValueError('Invalid OSTI mode selected')
                
            # start both handlers concurrently
            results = await asyncio.gather(api.run(), self.__search_coroutine(api.results, num_papers)) 
            results = results[1]  # api.run() does not return anything, get the actual results at index 1
            
        return results
    
    
    ### Helpers

            
    def __generate_func_dict(self, prefix):
        return {
            self.__strip_prefix(name, prefix): getattr(self, name) for name in dir(self)
            if self.__is_valid_function(name, prefix)
        }
    
    def __strip_prefix(self, name, prefix):
        return name[len(prefix):]

    
    def __is_valid_function(self, name, prefix):
        return (name.startswith(prefix) and callable(getattr(self, name))) and \
                self.__strip_prefix(name, prefix) in OSTI.MODES
                
    
    async def __search_coroutine(self, queue, num_papers):
        paper_ids = set()
        use_pbar = False
        if self.verbose and num_papers is not None:
            pbar = tqdm(total=num_papers)
            use_pbar = True
        while True:
            try:
                result = await queue.get()
                if result is None:
                    if use_pbar:
                        pbar.n = pbar.total
                        pbar.update(0)
                    break
                
                op, pid, paper = result
                if use_pbar:
                    self.pbar_count += 1
                    pbar.update(1)
                
                paper_ids.add(pid)
                if paper is not None:
                    self.__save(result)
                
                queue.task_done()  # signal that the queue item has been processed
            except asyncio.CancelledError:
                if use_pbar:
                    pbar.close()
                return list(paper_ids)
            
        if use_pbar:
            pbar.close()
        return list(paper_ids)
            
        
    async def __count_coroutine(self, queue):
        count = 0
        while True:
            try:
                result = await queue.get()
                if result is None:
                    break
                
                _, _, num = result
                count += num
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
            raise ValueError(f'The path "{path}" is not a directory')
        return {f.stem for f in path.glob('*.json')}
    
    def __save(self, data):
        save_func = self.save_funcs[self.mode]
        return save_func(data)
    
    def _save_fs(self, data):
        op, osti_id, paper = data
        with open(os.path.join(self.name, f'{osti_id}.json'), 'w') as fh:
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