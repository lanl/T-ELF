import httpx
import random
import asyncio
import warnings
import urllib

from ..utils import multi_urljoin, get_query_param


class OSTIApi:
    """
    Asynchronous OSTI API Handler.
    See https://www.osti.gov/api/v1/docs for details on the OSTI API

    This class provides methods for retrieving scientific literature asynchronously with the OSTI API. 
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
    BASE_URL = 'https://www.osti.gov/api/v1/records/'
    BASE_API_RATE = 15  # approximately how many times to call API per second
    QUERY_LIMIT = 100  # max number of papers per query search page
    

    def __init__(self, client: httpx.AsyncClient, key: str | None = None, *, ignore: set = set(), max_retries: int = 5):
        """
        Initialize the OSTIApi
        
        Parameters:
        -----------
        client: httpx.AsyncClient
            The asynchronous HTTP client that will be used for making requests
        ignore: set
            An optional set of OSTI paper IDs that will be ignored and not downloaded. This field can be 
            used to reduce the number of calls made to the API. If an OSTI paper has already been downloaded,
            it can be "ignored" so that it is not downloaded in a separate call to the API.
        key: str, None
            An optional OSTI API key. If provided, it allows for higher API rate limits. Defaults to None.
        max_retries: int 
            Maximum number of retries for a task or operation. Defaults to 5.
        
        Attributes:
        -----------
        client: httpx.AsyncClient
            An asynchronous HTTP client for making requests.
        key: str, None
            The API key provided during initialization.
        ignore: set
            Set of OSTI paper ids to ignore. Note that if the user tries to download a paper (id) that is in
            `ignore`, no call will be made to the OSTI API and instead a None will be placed
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
            a higher level class can wrap on OSTIApi to store the data at the required scale
        max_retries: int 
            Maximum number of retries for a task or operation
        """
        self.client = client
        self.key = key
        self.ignore = ignore
        self.todo = asyncio.Queue()
        self.seen = set()
        self.rate_limiter = asyncio.Event()  # rate limit lock, only unset when API returns code 429
        self.rate_limiter.set()
        self.semaphore = asyncio.Semaphore(self.BASE_API_RATE)
        self.num_workers = 25
        self.results = asyncio.Queue()  # queue for storing results
        self.max_retries = max_retries
    
    
    async def run(self):
        """
        Start multiple worker coroutines and await their completion.
        
        This coroutine initializes a number of worker coroutines based on `self.num_workers`. 
        Each worker is tasked with processing items (in this case urls to the OSTI API)
        
        After starting the workers, this coroutine waits for the 'todo' queue to be emptied 
        (or all tasks to be marked as done). Once the 'todo' queue is emptied, all worker tasks 
        are cancelled, signaling them to stop their work. 
        
        Finally, a sentinel value `None` is put into the `results` queue, indicating the end 
        of results.
        
        Returns:
        --------
        None
        """
        workers = [
            asyncio.create_task(self.worker())
            for _ in range(self.num_workers)
        ]
        await self.todo.join()
        for worker in workers:
            worker.cancel()
        await self.results.put(None)


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

        try:
            await self.make_request(item, op, url)
        except Exception as e:
            if retries < self.max_retries:
                wait = (2 ** retries) + random.uniform(0, 1)  # exponential backoff with jitter
                if not isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError)):
                    warnings.warn(f'[OSTI API]: An unexpected error occurred with '
                                  f'URL {url}: {e.__class__.__name__}, {e}', RuntimeWarning)
                
                if isinstance(e, ConnectionRefusedError):
                    self.rate_limiter.clear()
                
                await asyncio.sleep(wait)  # wait before retrying
                
                if isinstance(e, ConnectionRefusedError):
                    self.rate_limiter.set()
                await self.todo.put((item, op, url, retries + 1))  # add the URL back to the queue with incremented retry count
            else:
                warnings.warn(f'[OSTI API]: Max retries reached for URL {url}, '
                              f'skipping', RuntimeWarning)
            self.todo.task_done()
        else:
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
            - The operation (`op`) parameter determines the type of OSTI endpoint being used. If op == 'paper'
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
            If the server returns paper id but results do not contain OSTI paper id
        """
        async with self.semaphore:  # acquire semaphore
            await self.rate_limiter.wait()
            timeout = httpx.Timeout(10.0, read=20.0)  # 20 seconds to read, 10 seconds everything else
            response = await self.client.get(url, timeout = timeout,
                                              headers={'Accept': 'application/json'})
            
            if response.status_code == 200:
                response_json = response.json()
            elif response.status_code == 429:
                raise ConnectionRefusedError('Hit the OSTI API Rate Limit')
            elif response.status_code == 404:
                return  # paper not found 
            else:
                warnings.warn(f'[OSTI API]: Server returned unexpected code {response.status_code} ' \
                              f'with message {response.json()}', RuntimeWarning)
                response_json = {}

                
            if op == 'number':
                total = response.headers.get('X-Total-Count')
                if total is None:
                    total = 0
                else:
                    total = int(total)
                await self.results.put((op, None, total))    
            elif op == 'paper':
                response_json = response_json[0]
                osti_id = response_json.get('osti_id')
                if osti_id is None:
                    raise RuntimeError(f'No contents to save for paper {item}!')
                await self.results.put((op, osti_id, response_json))  # put JSON response in queue
            elif op == 'query':
                n = int(get_query_param(url, 'n'))
                query = get_query_param(url, 'q')
                page = int(get_query_param(url, 'page'))
                total = int(response.headers.get('X-Total-Count'))
                if n == 0:
                    n = total
                
                paperIds = [x['osti_id'] for x in response_json if 'osti_id' in x]
                paperIds = paperIds[:(n+self.QUERY_LIMIT)-(page*self.QUERY_LIMIT)]  # get as many papers as possible 
                if page * self.QUERY_LIMIT < min(n, total):
                    await self.find_papers_by_query(query, n, page=page+1)
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
            self.seen.add(url)
            await self.todo.put((item, op, url, 0))  # id, type of op, API endpoint, initial retry value

                            
    async def find_papers_by_id(self, data: list[str]):
        """
        Asynchronously prepares search URLs based on provided paper IDs and adds them to the processing queue.
        
        Constructs search URLs using OSTI paper IDs. The constructed URLs are then added to the processing 
        queue to be downloaded.
        
        Parameters:
        -----------
        data: list[str]: 
            A list of paper IDs to search for.
        
        Returns:
        --------
        None
        
        Raises:
        -------
            ValueError: If any of the provided search fields are not part of the allowed search fields.
        """        
        for item in data:
            if item in self.ignore:
                await self.results.put(('paper', item, None))
            else:
                url_parts = [self.BASE_URL, item]
                search_url = multi_urljoin(*url_parts)
                await self.add_to_queue(item, 'paper', search_url)
            
            
    async def find_papers_by_query(self, query: str, n: int = 1000, *, page: int = 1, number: bool = False):
        """
        Asynchronously prepares URLs based on some search query and adds them to the processing queue.

        Parameters:
        -----------
        query: str
            A query to search for
        n: int
            The maximum number of papers to return
        page: int
            By how many pages the search results should be offset. This parameter should really only
            be used by calls from `self.make_request()` and in 99.9% of cases can be ignored by the user.
        number: bool
            This is a boolean intended to be used by a higher level OSTI API handler to determine if the 
            number of papers in the search should be found rather than the papers themselves.
        
        Returns:
        --------
        None
        """
        search_query = urllib.parse.urlencode({'q': query})
        search_limit = urllib.parse.urlencode({'rows': self.QUERY_LIMIT})  # 100 is the max return limit 
        search_page = urllib.parse.urlencode({'page': page})
        search_n = urllib.parse.urlencode({'n': n})
        search_url = self.BASE_URL + '?' + search_query + '&' + search_page + '&' + search_limit + '&' + search_n
        
        if number:
            await self.add_to_queue(query, 'number', search_url)
        else:
            await self.add_to_queue(query, 'query', search_url)
