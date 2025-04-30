import httpx
import random
import asyncio
import requests
import warnings
import urllib

from ..utils import multi_urljoin, get_query_param

class SemanticScholarAPI:
    """
    Asynchronous Semantic Scholar (S2) API Handler.
    See https://api.semanticscholar.org/api-docs for details on the S2 API
    See https://github.com/allenai/s2-folks/blob/main/API_RELEASE_NOTES.md for current updates

    This class provides methods for retrieving scientific literature asynchronously with the S2 API. 
    It uses coroutines to efficiently make and manage multiple API requests concurrently. The class maintains 
    a task queue for pending requests, ensures rate limits are not exceeded, and has capabilities for URL deduplication 
    and exponential backoff retries. The retrieved papers are stored in an asynchronous queue and can be downloaded
    using a higher level handler. 

    Key Features:
    - Asynchronous handling of API requests using httpx.
    - Efficient rate limiting using asyncio semaphores.
    - Exponential backoff with jitter for retrying failed requests.
    - Retrieving papers through paper ids, author ids, or queries
    
    Note:
    This class should be used within an asynchronous context to fully leverage its capabilities.
    """
    BASE_URL = 'https://api.semanticscholar.org'
    #KEY_API_RATE = 100  # S2 API with authenticated key allows up to 100 calls per second
    
    # As of March 2024, the rate limit for authenticated users has been reduced
    # The documentation / blog posts do not state the new rate but previous rate of 100
    # causes SemanticScholarAPI to constantly bounce off of the rate limiter. The KEY_API_RATE
    # is being set to the BASE_API_RATE until further information is known.
    KEY_API_RATE = 15  
    
    BASE_API_RATE = 15  # S2 API allows 5000 calls in 5 minutes (~15 calls per second) without key
    QUERY_LIMIT = 100  # max number of papers per query search page
    SEARCH_FIELDS = {
        'paper': {'url', 'title', 'venue', 'publicationVenue', 'year', 'authors',
                  'externalIds', 'abstract', 'referenceCount', 'citationCount',
                  'influentialCitationCount', 'isOpenAccess', 'openAccessPdf',
                  'fieldsOfStudy', 's2FieldsOfStudy', 'publicationTypes', 
                  'publicationDate', 'journal', 'citationStyles', 'embedding',
                  'citations.paperId', 'references.paperId'}    
    }
    
    def __init__(self, client: httpx.AsyncClient, key: str | None = None, *, ignore: set = set(), max_retries: int = 5):
        """
        Initialize the SemanticScholarAPI
        
        Parameters:
        -----------
        client: httpx.AsyncClient
            The asynchronous HTTP client that will be used for making requests
        ignore: set
            An optional set of S2 paper IDs that will be ignored and not downloaded. This field can be 
            used to reduce the number of calls made to the API. If an S2 paper has already been downloaded,
            it can be "ignored" so that it is not downloaded in a separate call to the API.
        key: str, None
            An optional S2 API key. If provided, it allows for higher API rate limits. Defaults to None.
        max_retries: int 
            Maximum number of retries for a task or operation. Defaults to 5.
        
        Attributes:
        -----------
        client: httpx.AsyncClient
            An asynchronous HTTP client for making requests.
        key: str, None
            The API key provided during initialization.
        ignore: set
            Set of S2 paper ids to ignore. Note that if the user tries to download a paper (id) that is in
            `ignore`, no call will be made to the S2 API and instead a None will be placed
            in the results queue for the given id.
        num_workers: int
            The number of worker coroutines.
        todo: asyncio.Queue
            Queue for tasks that need to be processed.
        seen: set 
            A set to keep track of API urls that have already been seen or processed.
        semaphore: asyncio.Semaphore
            A semaphore to manage rate limits. It uses the rate limit associated with the provided API key 
            (if any) or falls back to a base rate.
        results: asyncio.Queue 
            A queue for storing processed results. The intent behind making this an asyncio.Queue is so that
            a higher level class can wrap on SemanticScholarAPI to store the data at the required scale
        max_retries: int 
            Maximum number of retries for a task or operation
        """
        self.client = client
        self.key = key
        self.ignore = ignore
        self.__workers = []
        self.todo = asyncio.Queue()
        self.todo_lock = asyncio.Lock()
        self.seen = set()
        self.rate_limiter = asyncio.Event()  # rate limit lock, only unset when API returns code 429
        self.rate_limiter.set()
        self.semaphore = asyncio.Semaphore(self.KEY_API_RATE if self.key is not None else self.BASE_API_RATE)
        self.num_workers = 150 if self.key is not None else 25
        self.results = asyncio.Queue()  # queue for storing results
        self.max_retries = max_retries
        self.tasks_in_progress_count = 0
        
        
    @classmethod    
    def validate_key(cls, key: str) -> bool:
        """
        Validates the given API key by testing it with a known paper.
        
        Parameters:
        -----------
        key: st
            The API key to validate.
        
        Returns:
        --------
        bool: 
            True if the API key is valid, otherwise False.
        
        Raises:
        -------
        requests.exceptions.Timeout: 
            If the request to the known endpoint times out.
        requests.exceptions.ProxyError: 
            If there's a proxy error while making the request.
        """
        api_url = 'https://api.semanticscholar.org/graph/v1/paper/10.1038/nrn3241'
        
        try:
            resp = requests.get(api_url, headers = {'Accept': 'application/json', 'x-api-key': key}, timeout=10)
            status_code = resp.status_code

            if status_code == 403:
                return False
            elif status_code == 200:
                return True
            else:
                warnings.warn(f'[S2 API]: Server returned code {status_code} when trying to validate API key', RuntimeWarning)
                return False
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(f'Server timed out when trying to validate S2 API key!')
        except requests.exceptions.ProxyError:
            raise requests.exceptions.ProxyError(f'Could not reach Semantic Scholar due to Proxy Error!')
        
        
    async def run(self):
        """
        Start multiple worker coroutines and await their completion.
        
        This coroutine initializes a number of worker coroutines based on `self.num_workers`. 
        Each worker is tasked with processing items (in this case urls to the S2 API)
        
        After starting the workers, this coroutine waits for the 'todo' queue to be emptied 
        (or all tasks to be marked as done). Once the 'todo' queue is emptied, all worker tasks 
        are cancelled, signaling them to stop their work. 
        
        Finally, a sentinel value 0 is put into the `results` queue, indicating the end 
        of results.
        
        Returns:
        --------
        None
        """
        self.__workers = [
            asyncio.create_task(self.worker())
            for _ in range(self.num_workers)
        ]
        await self.todo.join()
        for worker in self.__workers:
            worker.cancel()
        self.__workers = []
        await self.results.put(0)


    async def worker(self):
        """
        Continuously processes tasks using the `process_one` coroutine until cancelled.
        
        This coroutine acts as a worker that repeatedly calls the `process_one` coroutine 
        to process tasks. If the worker is cancelled (e.g., if an external event stops it),
        it catches the `asyncio.CancelledError` and gracefully stops the processing.
        
        Raises:
        -------
        Any exceptions raised by `process_one`, except for `asyncio.CancelledError` 
        which is caught and handled by the worker to stop gracefully.
        
        Returns:
        --------
        None
        """
        while True:
            try:
                await self.process_one()
            except asyncio.CancelledError:
                return
            
            
    async def process_one(self):
        """
        Asynchronously process a single task from the 'todo' queue, making a request to the given URL.
        
        The task details include the item to be processed, the operation, the URL for the request,
        and the number of retries that have been attempted so far. If the request fails, this 
        coroutine will apply an exponential backoff with jitter strategy to retry the request,
        until the maximum number of retries (`self.max_retries`) is reached.
        
        Notes:
            - The task is considered done (with `self.todo.task_done()`) both in cases of success 
              or after reaching the max number of retries.
            - Errors like `httpx.ReadTimeout`, `httpx.ConnectTimeout`, `httpx.ConnectError`, and 
              `ConnectionRefusedError` lead to a retry with exponential backoff, unless max retries is reached.
              Other errors raise a warning and lead to a retry.
            - The exponential backoff delay is calculated as (2 ** retries) + a random jitter between 0 and 1.
            
        Parameters:
        -----------
        None
            The method fetches tasks internally from `self.todo` queue.
        
        Returns:
        --------
        None 
            The task is marked as done (striked from the todo queue) when completed
        """
        item, op, url, retries = await self.todo.get()
        self.tasks_in_progress_count += 1
        try:
            await self.make_request(item, op, url)
        except Exception as e:
            if retries < self.max_retries:
                wait = (2 ** retries) + random.uniform(0, 1)  # exponential backoff with jitter
                if not isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError)):
                    warnings.warn(f'[S2 API]: An unexpected error occurred with '
                                  f'URL {url}: {e.__class__.__name__}, {e}', RuntimeWarning)
                
                if isinstance(e, ConnectionRefusedError):
                    self.rate_limiter.clear()
                
                await asyncio.sleep(wait)  # wait before retrying
                
                if isinstance(e, ConnectionRefusedError):
                    self.rate_limiter.set()
                async with self.todo_lock:  # add the URL back to the queue with incremented retry count
                    await self.todo.put((item, op, url, retries + 1))  
            else:
                warnings.warn(f'[S2 API]: Max retries reached for URL {url}, '
                              f'skipping', RuntimeWarning)
        
        finally:
            self.tasks_in_progress_count -= 1
            self.todo.task_done()


    async def make_request(self, item: str, op: str, url: str):
        """
        Asynchronously make an HTTP GET request to the specified URL and store the resulting JSON data.
        
        This coroutine acquires a semaphore before making the request, ensuring that the number of
        concurrent requests remains under the defined rate limits. After fetching the response,
        it processes the status code and either fetches the JSON data or raises an appropriate error.
        The processed data is then put into the `self.results` queue.
        
        Notes:
            - The function relies on acquiring a semaphore to handle rate limits (`self.semaphore`).
            - If the response status code is 200 (OK), the JSON response is read and stored in `self.results`.
              If it's 429 (Too Many Requests), a `ConnectionRefusedError` is raised.
              For all other status codes, an empty dictionary is stored for the item in `self.results`.
            - The operation (`op`) parameter determines the type of S2 endpoint being used. If op == 'paper'
              the information for the given paper will be retrieved and stored in the `self.results` queue. 
              If op == 'author', the papers written by said author will be placed in the `self.todo` to be
              downloaded. Similarly, if op == 'query', the papers for the given search query will be found
              and placed in the todo queue. 
        
        Parameters:
        -----------
        item: str
            The identifier or name of the item being processed.
        op: str
            The operation being performed. This operation is in ['paper', 'author', 'query']. 
        url: str
            The URL endpoint to make the HTTP GET request.
        
        Returns:
        --------
        None
            Results are placed in the `self.results` queue
        
        Raises:
        -------
        ConnectionRefusedError: 
            If the status code indicates an API rate limit hit (HTTP 429). 
        RuntimeError:
            If the server returns paper id but results do not contain S2 paper id
        """
        async with self.semaphore:  # acquire semaphore
            await self.rate_limiter.wait()
            timeout = httpx.Timeout(10.0, read=20.0)  # 20 seconds to read, 10 seconds everything else
            response = await self.client.get(url, timeout = timeout,
                                              headers={'Accept': 'application/json', 'x-api-key': self.key or ''})
            
            if response.status_code == 200:
                response_json = response.json()
            elif response.status_code == 429:
                raise ConnectionRefusedError('Hit the Semantic Scholar API Rate Limit')
            elif response.status_code == 404:
                return  # paper not found 
            else:
                warnings.warn(f'[S2 API]: Server returned unexpected code {response.status_code} ' \
                              f'with message {response.json()}', RuntimeWarning)
                response_json = {}

                
            if op == 'number':
                total = response_json.get('total')
                if not total:
                    total = len(response_json.get('data', []))
                await self.results.put((op, None, total))    
            elif op == 'paper':
                s2_paperId = response_json.get('paperId')
                if s2_paperId is None:
                    raise RuntimeError(f'No contents to save for paper {item}!')
                await self.results.put((op, s2_paperId, response_json))  # put JSON response in queue
            elif op == 'author':
                paperIds = [x['paperId'] for x in response_json.get('data', []) if 'paperId' in x]
                if paperIds:
                    await self.find_papers_by_id(paperIds)
            elif op == 'query':
                n = int(get_query_param(url, 'n'))
                query = get_query_param(url, 'query')
                offset = response_json.get('next', 0)
                total = response_json.get('total', 0)

                paperIds = [x['paperId'] for x in response_json.get('data', []) if 'paperId' in x]
                paperIds = paperIds[:(n+self.QUERY_LIMIT)-offset]  # get as many papers as possible 
                if offset < min(n, total):
                    await self.find_papers_by_query(query, n, offset=offset)
                if paperIds:
                    await self.find_papers_by_id(paperIds)
            elif op == 'bulk_query':
                n = int(get_query_param(url, 'n'))
                query = get_query_param(url, 'query')
                token = response_json.get('token', '')
                offset = response_json.get('next', 0)
                total = response_json.get('total', 0)

                paperIds = [x['paperId'] for x in response_json.get('data', []) if 'paperId' in x]
                if offset < min(n, total):
                    await self.find_papers_by_query_bulk(query, n, offset=offset, token=token)
                if paperIds:
                    await self.find_papers_by_id(paperIds)
            
    
    async def add_to_queue(self, item: str, op: str, url: str):
        """
        Asynchronously add a new task to the `self.todo` queue for processing if it hasn't been seen before.
        
        This coroutine first checks if the given URL is already in the `self.seen` set. If the URL
        has not been seen (processed or added to the queue), it is added to the 'todo' queue for 
        processing, and also added to the 'seen' set to ensure that duplicate URLs are not processed 
        multiple times.
        
        Notes:
            - Tasks added to the 'todo' queue are in the format: (item, op, url, retry_count).
            - `retry_count` is initialized to 0 for all new tasks.
        
        Parameters:
        -----------
        item: str 
            The identifier or name of the item to be processed.
        op: str
            The operation being performed. This operation is in ['paper', 'author', 'query']. 
        url:
            The URL endpoint associated with the item.
        
        Returns:
        --------
        None
        """
        if url not in self.seen:
            async with self.todo_lock:
                self.seen.add(url)
                await self.todo.put((item, op, url, 0))  # id, type of op, API endpoint, initial retry value

            
    async def clear_queue(self):
        """
        Clears all items from the `todo` queue.

        This method retrieves all tasks from the queue and marks them as done. This is useful in scenarios 
        where, due to some error or exception, you want to abandon processing of all tasks in the queue.

        Returns:
        --------
        None
        """
        async with self.todo_lock:
            while not self.todo.empty():
                try:
                    task = self.todo.get_nowait()
                    self.todo.task_done()  # mark task as done so join works correctly.
                except asyncio.QueueEmpty: 
                    break  # queue has been emptied
                
                
    async def cleanup(self):
        """
        Clean up the worker tasks and ensure the queue is cleared.

        This method is designed to be called in exceptional situations, such as
        handling a KeyboardInterrupt, to guarantee a graceful shutdown. It clears
        any remaining tasks in the `todo` queue, cancels the worker tasks, and 
        waits for all workers to finish their current operations.

        Returns:
        --------
        None
        """
        await self.clear_queue()
        
        # mark in-progres tasks as done
        for _ in range(self.tasks_in_progress_count):
            self.todo.task_done()
        
        for worker in self.__workers:
            worker.cancel()
        await asyncio.gather(*self.__workers, return_exceptions=True)
        await self.client.aclose()
                            
            
    async def find_papers_by_id(self, data: list[str], fields: list[str] | None = None, *, total: int = 0):
        """
        Asynchronously prepares search URLs based on provided paper IDs and adds them to the processing queue.
        
        Constructs search URLs using paper IDs and the desired fields. If no specific fields are provided, 
        it defaults to using a set of predefined search fields. The constructed URLs are then added to 
        the processing queue to be downloaded.
        
        Notes:
            - Semantic Scholar supports multiple types of paper ids as input. The following are examples
              of currently supported ids and how to request them:
                S2ID: a Semantic Scholar ID, e.g. 649def34f8be52c8b66281af98ae884c09aef38b
                CorpusId:<id> - a Semantic Scholar numerical ID, e.g. CorpusId:215416146
                DOI:<doi> - a Digital Object Identifier, e.g. DOI:10.18653/v1/N18-3011
                ARXIV:<id> - arXiv.rg, e.g. ARXIV:2106.15928
                MAG:<id> - Microsoft Academic Graph, e.g. MAG:112218234
                ACL:<id> - Association for Computational Linguistics, e.g. ACL:W12-3903
                PMID:<id> - PubMed/Medline, e.g. PMID:19872477
                PMCID:<id> - PubMed Central, e.g. PMCID:2323736
        
        Parameters:
        -----------
        data: list[str]: 
            A list of paper IDs to search for.
        fields: list[str], None
            A list of fields to be retrieved for each data ID. If not provided, defaults to using a 
            predefined set of search fields.
        total: int
            The upper limit on the number of papers to find from this search. If set to 0, no such limit
            is used. The default is 0. 
        
        Returns:
        --------
        None
        
        Raises:
        -------
            ValueError: If any of the provided search fields are not part of the allowed search fields.
        """
        search_endpoint = "/graph/v1/paper/"
        if fields is None:
            search_fields = {'fields': ','.join(self.SEARCH_FIELDS['paper'])}
        else:
            search_fields = set(fields)
            if not search_fields.issubset(self.SEARCH_FIELDS['paper']):
                raise ValueError(f'[S2 API]: Invalid fields provided. Search fields must be in {self.SEARCH_FIELDS["paper"]}')
            else:
                search_fields = {'fields': ','.join(search_fields)}
        
        for item in data:
            if item in self.ignore:
                await self.results.put(('paper', item, None))
            else:
                url_parts = [self.BASE_URL, search_endpoint, item]
                search_url = multi_urljoin(*url_parts)
                search_url += '?' + urllib.parse.urlencode(search_fields)
                await self.add_to_queue(item, 'paper', search_url)


    async def find_papers_by_author(self, data: list[str], *, total: int = 0, number: bool = False):
        """
        Asynchronously prepares search URLs based on provided S2 author IDs and adds them to the processing 
        queue.
        
        Given a list of S2 author IDs, urls are constructed to find each authors papers. Note that this 
        implementation will only retrieve up to 1,000 papers for each author. The S2 API has the
        capability to retrieve up to 10,000 papers per author but in the great majority of cases up to
        1,000 papers should be more than enough. `self.make_request()` can be modified to make this
        happen.

        Parameters:
        -----------
        data: list[str]: 
            A list of S2 author IDs to search for.
        total: int
            The upper limit on the number of papers to find from this search. If set to 0, no such limit
            is used. The default is 0. 
        number: bool
            This is a boolean intended to be used by a higher level S2 API handler to determine if the 
            number of papers in the search should be found rather than the papers themselves.
        
        Returns:
        --------
        None
        """
        search_endpoint = "/graph/v1/author/"
        search_fields = urllib.parse.urlencode({'fields': 'paperId'})
        search_limit = urllib.parse.urlencode({'limit': 1000})
        for item in data:
            url_parts = [self.BASE_URL, search_endpoint, f'{item}/papers']
            search_url = multi_urljoin(*url_parts)
            search_url += '?' + search_fields + '&' + search_limit
            if number:
                await self.add_to_queue(item, 'number', search_url)
            else:
                await self.add_to_queue(item, 'author', search_url)
            
            
    async def find_papers_by_query(self, query: str, n: int = 1000, *, offset: int = 0, number: bool = False):
        """
        Asynchronously prepares URLs based on some search query and adds them to the processing queue.
        
        Given a query (title, topic, keywords, etc), urls are constructed to find corresponding papers.
        In many cases a query will yield far more papers than can be processed. For example using
        "tensor decomposition" will produce over 1.1 million papers. For this reason, the n parameter
        can be used to limit the number of papers retrieved. Fortunately, Semantic Scholar ranks the
        results so the most relevant papers will be provided first.

        Parameters:
        -----------
        query: str
            A query to search for
        n: int
            The maximum number of papers to return
        offset: int
            By how many papers the search results should be offset. This parameter should really only
            be used by calls from `self.make_request()` and in 99.9% of cases can be ignored by the user.
        number: bool
            This is a boolean intended to be used by a higher level S2 API handler to determine if the 
            number of papers in the search should be found rather than the papers themselves.
        
        Returns:
        --------
        None
        """
        search_endpoint = "/graph/v1/paper/search"
        search_query = urllib.parse.urlencode({'query': query})
        search_limit = urllib.parse.urlencode({'limit': self.QUERY_LIMIT})  # 100 is the max return limit 
        search_offset = urllib.parse.urlencode({'offset': offset})
        search_n = urllib.parse.urlencode({'n': n})
        search_url = urllib.parse.urljoin(self.BASE_URL, search_endpoint) + '?' + search_query \
                     + '&' + search_offset + '&' + search_limit + '&' + search_n
        
        if number:
            await self.add_to_queue(query, 'number', search_url)
        else:
            await self.add_to_queue(query, 'query', search_url)

            
    async def find_papers_by_query_bulk(self, query: str, n: int = 0, *, offset: int = 0, 
                                              token: str = '', number: bool = False):
        """
        Asynchronously prepares URLs based on some search query and adds them to the processing queue.
        
        Given a query (title, topic, keywords, etc), urls are constructed to find corresponding papers.
        This specific endpoint differs from `find_papers_by_query` in that it utilizes the new (introduced
        December 2023) bulk search endpoint. This allows for complex text search queries to be built 
        and greater quantities of papers to be retrieved. The tradeoff is that this endpoint does not 
        employ the S2 graph relevance ranking for sorting the results. The text query will be matched 
        against the document's title and abstract. All terms are stemmed in English. By default all terms 
        in the query must be present in the paper.

        The match query supports the folowing syntax:

        + for AND operation
        | for OR operation
        - negates a term
        " collects terms into a phrase
        * can be used to match a prefix
        ( and ) for precedence
        ~N after a word matches within the edit distance of N (Defaults to 2 if N is omitted)
        ~N after a phrase matches with the phrase terms separated up to N terms apart (Defaults to 2 if N is omitted)

        Examples:
        fish ladder matches papers that contain "fish" and "ladder"
        fish -ladder matches papers that contain "fish but not "ladder"
        fish | ladder mathces papers that contain "fish" or "ladder"
        "fish ladder" mathces papers that contain the phrase "fish ladder"
        (fish ladder) | outflow matches papers that contain "fish" and "ladder" OR "outflow"
        fish~ matches papers that contain "fish", "fist", "fihs", etc.
        "fish ladder"~3 mathces papers that contain the phrase "fish ladder" or "fish is on a ladder"
        

        Parameters:
        -----------
        query: str
            A query to search for
        n: int
            The maximum number of papers to return
        offset: int
            By how many papers the search results should be offset. This parameter should really only
            be used by calls from `self.make_request()` and can be ignored by the user.
        token: str
            A search token returned by the bulk endpoint. Used for chaining search queries together (similar to
            the numeric offset used for `find_papers_by_query`
        number: bool
            This is a boolean intended to be used by a higher level S2 API handler to determine if the 
            number of papers in the search should be found rather than the papers themselves.
        
        Returns:
        --------
        None
        """
        search_endpoint = "/graph/v1/paper/search/bulk"
        search_query = urllib.parse.urlencode({'query': query})
        search_token = urllib.parse.urlencode({'token': token})
        search_offset = urllib.parse.urlencode({'offset': offset})
        search_n = urllib.parse.urlencode({'n': n})
        search_url = urllib.parse.urljoin(self.BASE_URL, search_endpoint) + '?' + search_query + '&' + search_offset
        if token:
            search_url = search_url + '&' + search_token
        search_url = search_url + '&' + search_n
        
        if number:
            await self.add_to_queue(query, 'number', search_url)
        else:
            await self.add_to_queue(query, 'bulk_query', search_url)
            
            
    # GETTERS / SETTERS


    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, key):
        if key is not None and (not key or not self.validate_key(key)):
            raise ValueError(f'[S2 API]: The key "{key}" was not accepted by the Semantic Scholar API')
        self._key = key
