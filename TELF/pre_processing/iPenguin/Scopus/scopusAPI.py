import sys
import httpx
import urllib
import random
import asyncio
import requests
import warnings

from ..utils import get_from_dict, multi_urljoin, get_human_readable_timestamp


class ScopusAPI:
    """
    Asynchronous Scopus API Handler.
    See https://dev.elsevier.com for details on the Scopus API

    This class provides methods for retrieving scientific literature asynchronously with the Scopus API. 
    It uses coroutines to efficiently make and manage multiple API requests concurrently. The class maintains 
    a task queue for pending requests, ensures rate limits are not exceeded, and has capabilities for URL deduplication 
    and exponential backoff retries. The retrieved papers are stored in an asynchronous queue and can be downloaded
    using a higher level handler. 

    Key Features:
    - Asynchronous handling of API requests using httpx.
    - Efficient rate limiting using asyncio semaphores.
    - Exponential backoff with jitter for retrying failed requests.
    - Retrieving papers through paper ids, authors, titles, and other queries
    
    Note:
    This class should be used within an asynchronous context to fully leverage its capabilities.
    """
    BASE_URL = 'https://api.elsevier.com'
    
    # Scopus API supports variable API rates for different end points
    # This API Handler uses a static API rate for all endpoints
    # The default is set to the most common rate seen on Scopus
    # See https://dev.elsevier.com/api_key_settings.html for more details
    API_RATE = 2
    
    # Set a verbosity level that will trigger debug print
    DEBUG_MODE = 100
    
    def __init__(self, 
                 client: httpx.AsyncClient, 
                 key: str, 
                 *, 
                 ignore: set = set(), 
                 max_retries: int = 5,
                 verbose: bool | int = False):
        """
        Initialize the ScopusAPI
        
        Parameters:
        -----------
        client: httpx.AsyncClient
            The asynchronous HTTP client that will be used for making requests
        key: str
             The Scopus API key
        ignore: set
            An optional set of Scopus paper IDs that will be ignored and not downloaded. This field can be 
            used to reduce the number of calls made to the API. If a Scopus paper has already been downloaded,
            it can be "ignored" so that it is not downloaded in a separate call to the API.
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
        __workers: List[asyncio.Task] 
            A list of worker tasks
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
        self.verbose = verbose
        self.client = client
        self.key = key
        self.ignore = ignore
        self.__workers = []
        self.num_workers = self.API_RATE
        self.todo = asyncio.Queue()
        self.todo_lock = asyncio.Lock()
        self.rate_limiter = asyncio.Event()  # rate limit lock, only unset when API returns code 429
        self.rate_limiter.set()
        self.semaphore = asyncio.Semaphore(self.API_RATE)
        self.results = asyncio.Queue()  # queue for storing results
        self.max_retries = max_retries
        
        
    @classmethod    
    def validate_key(cls, key, verbose=False) -> bool:
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
        api_url = f'{cls.BASE_URL}/content/abstract/scopus_id/85148250027'
        
        try:
            resp = requests.get(api_url, headers = {'Accept': 'application/json', 'X-ELS-APIKey': key}, timeout=10)
            status_code = resp.status_code

            if status_code in {401, 403}:
                return False
            elif status_code == 200:
                pass
            elif status_code == 429:                
                error_message = get_from_dict(resp.json(), ['error-response', 'error-code'])
                if error_message == 'TOO_MANY_REQUESTS':
                    warnings.warn(f'[Scopus API]: Request has been placed in time-out for exceeding ' \
                                  f'quota or rate limits for "{key}"', RuntimeWarning)
            else:
                warnings.warn(f'[Scopus API]: Server returned code {status_code} when trying to validate API key', RuntimeWarning)
                return False
            
            quota = resp.headers.get('X-RateLimit-Remaining')
            reset_epoch = resp.headers.get('X-RateLimit-Reset')
            reset_epoch = get_human_readable_timestamp(int(reset_epoch)) \
                          if reset_epoch else 'undetermined'
            
            if verbose and quota is not None:
                print(f'[Scopus API]: Remaining API calls: {quota:<4}\n'
                      f'              Quota resets at: {reset_epoch:>23}\n', file=sys.stderr)              
            return True
            
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(f'Server timed out when trying to validate Scopus API key!')
        except requests.exceptions.ProxyError:
            raise requests.exceptions.ProxyError(f'Could not reach Scopus due to Proxy Error!')


    @classmethod    
    def get_quota(cls, key) -> bool:
        """
        Get the quota for a given API key by testing it with a known paper.
        
        Parameters:
        -----------
        key: st
            The API key for which to get the quota
        
        Returns:
        --------
        int: 
            The quota
        
        Raises:
        -------
        requests.exceptions.Timeout: 
            If the request to the known endpoint times out.
        requests.exceptions.ProxyError: 
            If there's a proxy error while making the request.
        ValueError:
            If the key is invalid
        """
        api_url = f'{cls.BASE_URL}/content/abstract/scopus_id/85148250027'
        
        try:
            resp = requests.get(api_url, headers = {'Accept': 'application/json', 'X-ELS-APIKey': key}, timeout=10)
            status_code = resp.status_code

            if status_code in {200, 429}:           
                quota = int(resp.headers.get('X-RateLimit-Remaining', -1))
                return quota
            elif status_code in {401, 403}:
                raise ValueError('Invalid Scopus key')
            else:
                raise ValueError(f'Server returned code {status_code} when trying to get quota for API key')
            
        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(f'Server timed out when trying to get quota for Scopus API key!')
        except requests.exceptions.ProxyError:
            raise requests.exceptions.ProxyError(f'Could not reach Scopus due to Proxy Error!')
            
            
    async def run(self):
        """
        Start multiple worker coroutines and await their completion.
        
        This coroutine initializes a number of worker coroutines based on `self.num_workers`. 
        Each worker is tasked with processing items (in this case urls to the Scopus API)
        
        After starting the workers, this coroutine waits for the 'todo' queue to be emptied 
        (or all tasks to be marked as done). Once the 'todo' queue is emptied, all worker tasks 
        are cancelled, signaling them to stop their work. 
        
        Finally, a sentinel value `None` is put into the `results` queue, indicating the end 
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
            except ConnectionAbortedError:
                await self.clear_queue()
                await self.results.put(127)
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
        item, op, url, data, retries = await self.todo.get()

        try:
            await self.make_request(item, op, url, data)
        except Exception as e:
            if isinstance(e, ConnectionAbortedError):
                print(f'[Scopus API]: {e}', file=sys.stderr)
                self.todo.task_done()
                raise ConnectionAbortedError('Hit Quota')
                
            if retries < self.max_retries:
                wait = (2 ** retries) + random.uniform(0, 1)  # exponential backoff with jitter
                if not isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError)):
                    warnings.warn(f'[Scopus API]: An unexpected error occurred with URL ' \
                                  f'{url}: {e.__class__.__name__}, {e}', RuntimeWarning)
                if isinstance(e, ConnectionRefusedError):
                    self.rate_limiter.clear()
                
                await asyncio.sleep(wait)  # wait before retrying
                
                if isinstance(e, ConnectionRefusedError):
                    self.rate_limiter.set()
                async with self.todo_lock:
                    await self.todo.put((item, op, url, data, retries + 1))  # add the URL back to the queue with incremented retry count
            else:
                warnings.warn(f'[Scopus API]: Max retries reached for URL ' \
                              f'{url}, skipping', RuntimeWarning)
            self.todo.task_done()
        else:
            self.todo.task_done()


    async def make_request(self, item: str, op: str, url: str, data: dict):
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
            response = await self.client.post(url, data = data, timeout = timeout,
                                              headers={'Accept': 'application/json', 'X-ELS-APIKey': self.key})
            if response.status_code == 200:
                response_json = response.json()
            elif response.status_code == 429:
                reset_epoch = response.headers.get('X-RateLimit-Reset')
                reset_epoch = get_human_readable_timestamp(int(reset_epoch)) \
                              if reset_epoch else 'undetermined'
                
                if response.headers.get('X-RateLimit-Remaining') == '0':
                    raise ConnectionAbortedError(f'API Key exceeded quota. Quota resets at {reset_epoch}')
                else:
                    raise ConnectionRefusedError('Hit the Scopus API Rate Limit')
            elif response.status_code == 400:
                raise ValueError(f'The query "{data.get("query")}" is invalid')
            else:
                warnings.warn(f'[Scopus API]: Server returned unexpected code {response.status_code}', RuntimeWarning)
                response_json = {}
            
            if op == 'paper':
                search_results = response_json.get('abstracts-retrieval-response', {})
                eid = get_from_dict(search_results, ['coredata', 'eid'])
                await self.results.put((op, eid, search_results))  # put JSON response in queue
            elif op == 'number':
                total = get_from_dict(response_json, ['search-results', 'opensearch:totalResults'])
                if not total:
                    total = 0
                await self.results.put((op, None, total))    
            elif op == 'query':
                n = data['n']
                query = data['query']
                local_count =  data['local_count']
                
                search_results = response_json.get('search-results', {})
                itemsPerPage = len(search_results.get('entry', []))
                cursor =  search_results.get('cursor', {}).get('@next')
                total = int(search_results.get('opensearch:totalResults', 0))
                if n == 0:
                    n = total
                    
                paperIds = [x['prism:url'] for x in search_results.get('entry', {}) if 'prism:url' in x]
                paperIds = [urllib.parse.urlparse(x).path.split('/')[-1] for x in paperIds]                
                if isinstance(self.verbose, int) and self.verbose >= self.DEBUG_MODE:  # debug print statement
                    print(f'\n[DEBUG]: Processed {local_count} papers, fetching {itemsPerPage} new papers\n'
                          f'         Requesting up to {n} documents from {total} search results', file=sys.stderr)
                    if self.verbose > self.DEBUG_MODE * 10: # max debug mode; dump everything
                        for pid in paperIds:
                            print(f'2-s2.0-{pid}', file=sys.stderr)
                
                if cursor != '*' and local_count < min(n, total):
                    await self.find_papers_by_query(query, n, count=min(local_count + itemsPerPage, n), cursor=cursor)
                if paperIds:
                    await self.find_papers_by_id(paperIds[:n - local_count])  # make sure n limiter is respected) 
                    
    
    async def add_to_queue(self, item: str, op: str, url: str, data: dict):
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
        async with self.todo_lock:
            await self.todo.put((item, op, url, data, 0))  # id, type of op, API endpoint, initial retry value

        
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
        for worker in self.__workers:
            worker.cancel()
        await asyncio.gather(*self.__workers, return_exceptions=True)
        await self.client.aclose()
        
                
    async def find_papers_by_id(self, data: list[str], fields: list[str] | None = None):
        search_endpoint = "/content/abstract/scopus_id"
        for item in data:
            if item in self.ignore:
                await self.results.put(('paper', f'2-s2.0-{item}', None))
            else:
                url_parts = [self.BASE_URL, search_endpoint, item]
                search_url = multi_urljoin(*url_parts)
                await self.add_to_queue(item, 'paper', search_url, {})
        
        
    async def find_papers_by_query(self, query: str, n: int = 1000, *, count: int = 0, cursor: str = '*'):
        search_endpoint = "/content/search/scopus"
        search_fields = {
            'field': 'prism:url',
            'query': query,
            'cursor': cursor,
            'count': 200,
            'local_count': count,
            'n': n,
        }
        
        url_parts = [self.BASE_URL, search_endpoint]
        search_url = multi_urljoin(*url_parts)
        await self.add_to_queue(query, 'query', search_url, search_fields)

        
    async def get_num_papers_from_query(self, query: str):
        search_endpoint = "/content/search/scopus"
        search_fields = {
            'field': 'prism:url',
            'query': query,
        }
        
        url_parts = [self.BASE_URL, search_endpoint]
        search_url = multi_urljoin(*url_parts)
        await self.add_to_queue(query, 'number', search_url, search_fields)
            
            
    # GETTERS / SETTERS


    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, key):
        if not self.validate_key(key, self.verbose):
            raise ValueError(f'The key "{key}" was not accepted by the Scopus API')
        self._key = key