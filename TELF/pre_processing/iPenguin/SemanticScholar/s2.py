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
from .s2API import SemanticScholarAPI
from ..utils import gen_chunks


def get_df_helper(files):
    data = {
        's2id': [],
        'doi': [],
        'year': [],
        'title': [],
        'abstract': [],
        's2_authors': [],
        's2_author_ids': [],
        'citations': [],
        'references': [],
        'num_citations': [],
        'num_references': [],
    }

    for f in files:
        with open(f, 'r') as fh:
            contents = json.load(fh)

        s2id = None
        if contents is not None and 'paperId' in contents:
            s2id = contents['paperId']
        data['s2id'].append(s2id)

        doi = None
        if contents is not None and 'externalIds' in contents:
            doi = contents['externalIds'].get('DOI')
        data['doi'].append(doi)

        title = None
        if contents is not None and 'title' in contents:
            title = contents['title']
        data['title'].append(title)

        abstract = None
        if contents is not None and 'abstract' in contents:
            abstract = contents['abstract']
        data['abstract'].append(abstract)

        year = None
        if contents is not None and 'year' in contents:
            year = contents['year']
        data['year'].append(year)

        authors = None
        if contents is not None and 'authors' in contents:
            authors = [str(x.get('name', None)) for x in contents['authors']]
            authors = [x for x in authors if x is not None]
            if not authors:
                authors = None
            else: 
                authors = ';'.join(authors)
        data['s2_authors'].append(authors)

        author_ids = None
        if contents is not None and 'authors' in contents:
            author_ids = [str(x.get('authorId', None)) for x in contents['authors']]
            author_ids = [x for x in author_ids if x is not None]
            if not author_ids:
                author_ids = None
            else: 
                author_ids = ';'.join(author_ids)
        data['s2_author_ids'].append(author_ids)

        citations = None
        num_citations = 0
        if contents is not None and 'citations' in contents:
            citations = [str(x.get('paperId', None)) for x in contents['citations']]
            citations = [x for x in citations if x is not None]
            if not citations:
                citations = None
            else: 
                num_citations = len(citations)
                citations = ';'.join(citations)
        data['citations'].append(citations)
        data['num_citations'].append(num_citations)
        
        references = None
        num_references = 0
        if contents is not None and 'references' in contents:
            references = [str(x.get('paperId', None)) for x in contents['references']]
            references = [x for x in references if x is not None]
            if not references:
                references = None
            else:
                num_references = len(references)
                references = ';'.join(references)
        data['references'].append(references)
        data['num_references'].append(num_references)
        
    # turn into dictionary and return
    df = pd.DataFrame.from_dict(data)
    return df


class SemanticScholar:
    
    MODES = {'fs'}
    
    # Semantic Scholar has a search limit on how many papers can be acquired through query
    SEARCH_LIMIT = 1000
    BULK_SEARCH_LIMIT = 1000000
    
    def __init__(self, 
                 key: str | None = None, 
                 mode: str = 'fs', 
                 name: str = 's2', 
                 n_jobs: int = -1, 
                 ignore = None,
                 verbose: bool = False):
        """
        Initializes the iPenguin.SemanticScholar instance with the specified key, mode, and optional settings.

        This constructor method sets up the SemanticScholar instance by initializing the key, mode, name, papers to 
        ignore, verbosity of output, and number of jobs for parallel processing (if applicable). It then attempts to 
        establish a connection to the S2 API to validate the key

        Parameters:
        -----------
        key: str, (optional)
            The API key to be used for this instance. If defined, this must be a valid API key. Can also be None
            to download papers at a slower rate with no key.
        mode: str, (optional)
            The mode in which the SemanticScholar instance should operate in. Currently one mode is supported, 'fs'. 
            This is the file system mode and will download papers to the directory path provided with `name`.
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
            of verbosity. Default is True.
        """
        self.key = key
        self.api = None
        self.mode = mode
        self.name = name
        self.n_jobs = n_jobs
        self.ignore = ignore
        self.verbose = verbose

        # create dictionaries of helper functions for supported modes
        prefixes = ['_prepare_', '_save_', '_read_']
        for p in prefixes:
            setattr(self, f"{p[1:-1]}_funcs", self.__generate_func_dict(p))
            
            
    @classmethod
    def get_df(cls, path, targets=None, *, backend='threading', n_jobs=1, verbose=False):
        path = pathlib.Path(path)
        if not path.is_dir():
            raise ValueError('The `path` parameter must be a path to an existing directory')
        if not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError('`n_jobs` must be a positive integer')
        
        if targets is not None:
            files = [file for name in targets for file in path.glob(f"{name}.json")]
        else:
            targets = []
            files = list(path.glob("*.json"))

        if not files:  # if no valid files found, terminate execution here by returning None
            return None, targets

        # chunk the files for parallel processing
        n_jobs = min(n_jobs, len(files))  
        chunks = gen_chunks(files, n_jobs)

        # generate dataframe slices in parallel
        jobs = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(delayed(get_df_helper)(c) for c in chunks)
        df = pd.concat(jobs, sort=False)
        df = df.reset_index(drop=True)
        return df, targets
    
    
    def count(self, data, mode, verbose=None):
        if verbose is None:
            verbose = self.verbose
        if verbose:
            start = time.perf_counter()
        
        num_papers =  asyncio.run(self.count_coroutine(data, mode), debug=False)
                
        if verbose:
            end = time.perf_counter()
            print(f"[S2]: Found {num_papers:,} papers in {end - start:.2f}s", file=sys.stderr)
        return num_papers
    
    
    async def count_coroutine(self, data, mode):
        async with httpx.AsyncClient() as client:
            api = SemanticScholarAPI(client, key=self.key)
            
            if mode == 'paper':
                return len(data)
            elif mode =='author':
                await api.find_papers_by_author(data, number=True)
            elif mode == 'query':
                await api.find_papers_by_query(data, number=True)
            elif mode == 'bulk':
                await api.find_papers_by_query_bulk(data, number=True)
            else:
                raise ValueError('Invalid S2 mode selected')
            count = await asyncio.gather(api.run(), self.__count_coroutine(api.results)) # start both handlers concurrently
            count = count[1]  # api.run() does not return anything, get the actual results at index 1
        return int(count)
    
    
    def search(self, data, mode, *, n=0):
        self.pbar_count = 0
        if self.verbose:
            start = time.perf_counter()
        
        if n == 0:
            n = self.BULK_SEARCH_LIMIT if mode == 'bulk' else self.SEARCH_LIMIT
        elif (n > self.BULK_SEARCH_LIMIT) or (mode != 'bulk' and n > self.SEARCH_LIMIT):
            n = self.SEARCH_LIMIT
            warnings.warn(f'[S2]: `n` exceeds the maximum allowed number of papers returned ' \
                          'by Semantic Scholar', RuntimeWarning)
        
        num_papers = self.count(data, mode, verbose=False)
        if num_papers == 0:
            raise ValueError('No S2 papers for `data` can be found. If you expect results for ' \
                             'this search, check your syntax')
        elif mode == 'query' and (n < num_papers):
            if n > 0:
                warnings.warn(f'[S2]: Found {num_papers} papers for query but limit is set ' \
                              'at {n}', RuntimeWarning)
            num_papers = n
        
        # prepare search
        ignore = self.__prepare()
        
        # search
        try:
            paper_ids = asyncio.run(self.search_coroutine(data, mode, ignore, n, num_papers), debug=False)
        except KeyboardInterrupt:
            asyncio.run(self.cleanup_coroutine(), debug=False)
            raise KeyboardInterrupt
            
        if self.verbose:
            end = time.perf_counter()
            time.sleep(1)  # if verbose, wait for all s2API messages to finish printing
            print(f"[S2]: Finished downloading {len(paper_ids):,} papers for given query in {end - start:.2f}s",  file=sys.stderr)
    
        # process
        df = self.__read(paper_ids)
        return df
    
    
    async def search_coroutine(self, data, mode, ignore, n, num_papers):
        if self.api is not None:
            await self.cleanup_coroutine()
        
        client = httpx.AsyncClient()
        self.api = SemanticScholarAPI(client, key=self.key, ignore=ignore)
            
        if mode == 'paper':
            n = len(data)
            num_papers = len(data)
            await self.api.find_papers_by_id(data)
        elif mode == 'author':
            await self.api.find_papers_by_author(data)
        elif mode == 'query':
            await self.api.find_papers_by_query(data, n=n)
        elif mode == 'bulk':
            await self.api.find_papers_by_query_bulk(data, n=n)
        else:
            raise ValueError('Invalid S2 mode selected')

        # start both handlers concurrently
        results = await asyncio.gather(self.api.run(), self.__search_coroutine(self.api.results, n, num_papers)) 
        results = list(results[1])  # api.run() does not return anything, get the actual results at index 1
        
        # terminate client
        await self.cleanup_coroutine()
        
        # return downloaded paper ids
        return results
    
    
    async def cleanup_coroutine(self):
        if self.api is not None:
            await self.api.cleanup()
            self.api = None
    
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
                self.__strip_prefix(name, prefix) in SemanticScholar.MODES
                
    
    async def __search_coroutine(self, queue, n, num_papers):
        use_pbar = bool(self.verbose)
        if use_pbar:
             pbar = tqdm(initial=0, total=num_papers)
        
        errno = 0
        found_papers = set()
        while True:
            try:
                result = await queue.get()
                if isinstance(result, int) or n <= self.pbar_count:
                    errno = result if isinstance(result, int) else 129
                    if errno == 129:  # early termination of search due to n
                        if self.verbose:
                            print(f"\n[S2]: Early termination of search after finding {n} papers. . .",  file=sys.stderr)
                        errno = 0  # correct error code back to successr
                    
                    # manage the progress bar 
                    if use_pbar:
                        pbar.close()
                    
                    queue.task_done()  # signal that the queue item has been processed
                    break
                
                op, pid, paper = result
                if pid not in found_papers:
                    self.pbar_count += 1
                    if use_pbar:
                        pbar.update(1)
                
                found_papers.add(pid)
                if paper is not None:
                    self.__save(result)
                
                queue.task_done()  # signal that the queue item has been processed
            except asyncio.CancelledError:
                return found_papers
            
        
        # terminate client
        await self.cleanup_coroutine()
        return found_papers
            
        
    async def __count_coroutine(self, queue):
        count = 0
        while True:
            try:
                result = await queue.get()
                if isinstance(result, int):
                    errno = result  # success if 0, otherwise error
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
            raise ValueError(f'The path {path!r} is not a directory')
        
        if self.ignore is not None:
            return self.ignore
        else:
            return {f.stem for f in path.glob('*.json')}
    
    def __save(self, data):
        save_func = self.save_funcs[self.mode]
        return save_func(data)
    
    def _save_fs(self, data):
        op, s2id, paper = data
        with open(os.path.join(self.name, f'{s2id}.json'), 'w') as fh:
            json.dump(paper, fh, indent=4)
            
    def __read(self, paper_ids):
        read_func = self.read_funcs[self.mode]
        return read_func(paper_ids)
    
    def _read_fs(self, paper_ids):
        return SemanticScholar.get_df(self.name, targets=paper_ids, n_jobs=self.n_jobs, verbose=self.verbose)

    
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