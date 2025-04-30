import re
import sys
import pathlib
import warnings
import pandas as pd
from tqdm import tqdm
from rbloom import Bloom
from joblib import Parallel, delayed

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

from ...helpers.host import verify_n_jobs
from ...helpers.data_structures import verify_attributes, transform_dict
from .utils import create_text_query
from .crocodile import process_scopus_json, process_scopus_xml, process_s2_json 
from .crocodile import form_scopus_df, form_s2_df, form_df

class Penguin:
    """
    A handler class for managing connections and operations with a documents MongoDB database.

    The purpose of this class it to create a software layer to easily store and query Scopus
    and S2 documents. Penguin provides functionality to connect to a MongoDB database, perform read 
    and write operations, and handle authentication for secure database interactions. It supports 
    operations such as adding data, retrieving data, and verifying database integrity.
    """
    # define the default collection names
    S2_COL = 'S2'
    SCOPUS_COL = 'Scopus'
    
    ## Define the default search settings
    # S2 and Scopus papers are defined 
    DEFAULT_ATTRIBUTES = {
        'S2': {
            'id': 'paperId',
            'doi': 'externalIds.DOI',
            'text': ['title', 'abstract'],
            'author_ids': ('authors', 'authorId'),
            'citations': 'citations',
            'references': 'references',
        },
        'Scopus': {
            'id': 'eid',
            'doi': 'doi',
            'text': ['title', 'abstract'],
            'author_ids': ('bibrecord.head.author-group.author', '@auid'),
            'citations': None,
            'references': None,
        },
    }

    
    def __init__(self, uri, db_name, username=None, password=None, n_jobs=-1, verbose=False):
        """
        Initializes the Penguin instance with the specified URI, database name, and optional settings.

        This constructor method sets up the Penguin instance by initializing the MongoDB URI, database name,
        verbosity of output, and number of jobs for parallel processing (if applicable). It then attempts to 
        establish a connection to the MongoDB database by calling Penguin._connect().

        Parameters:
        -----------
        uri: str
            The MongoDB URI used to establish a connection.
        db_name: str
            The name of the database to which the connection will be established.
        username: str, (optional)
            The username for the Mongo database. If None, will try to use DB without authentication. Default is None.
        password: str, (optional)
            The password for the Mongo database. If None, will try to use DB without authentication. Default is None.
        n_jobs: int, (optional)
            The number of jobs for parallel processing. Note that this setting is only 
            used for adding new data to the database and converting documents to SLIC
            DataFrame format for output. Default is -1 (use all available cores).
        verbose: bool, int (optional)
            If set to True, the class will print additional output for debugging or information 
            purposes. Can also be an int where verbose >= 1 means True with a higher integer
            controlling the level of verbosity. Default is True.

        Raises:
        -------
        ValueError:
            Attribute was given an invalid value
        TypeError:
            Attribute has an invalid type
        ConnectionFailure: 
            If the connection to the MongoDB instance fails during initialization.
        """
        self.uri = uri
        self.db_name = db_name
        self.username = username
        self.password = password
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.client = None
        self.db = None
        self._s2_attributes = self.DEFAULT_ATTRIBUTES[self.S2_COL]
        self._scopus_attributes = self.DEFAULT_ATTRIBUTES[self.SCOPUS_COL]
        self._connect()
        
    
    # 1. Connecting to the Penguin Database
        
        
    def _connect(self):
        """
        Establishes a connection to the MongoDB database and validates its structure.

        This method attempts to connect to a MongoDB instance using the MongoDB URI 
        and database name provided during the class initialization. It first tries 
        to establish a client connection to MongoDB and then selects the specified database. 
        The method then checks if the MongoDB server is reachable by issuing the 'ismaster' 
        command. If the command fails, it raises a ConnectionFailure exception.

        After establishing a successful connection, the method validates the database structure 
        by ensuring the presence of the Scopus and S2 collections. The database must contain
        these collections if using the database in read-only mode. 

        Raises:
        -------
        ConnectionFailure
            If the connection to MongoDB fails or the 'ismaster' command 
            cannot be executed, indicating a connection issue.
        ValueError
            If the database does not contain the required Scopus/S2 collections
        """
        try:
            
            # create the uri with authentication if provided
            uri = f"mongodb://{self.username}:{self.password}@{self.uri}" if self.username and self.password else f"mongodb://{self.uri}"
    
            # connect to DB
            self.client = MongoClient(uri)
            self.db = self.client[self.db_name]
            
            # check if connection is succesful
            try:
                self.client.admin.command('ismaster')
            except Exception as e:
                raise ConnectionFailure(f'Failed to connect to database {self.db_name}: {e}')
            
            # make sure databse has expected collections for Scopus and S2 documents
            if not self._validate_database([self.S2_COL, self.SCOPUS_COL]):
                raise ValueError('MongoDB connection successful, but the database is not valid.')
            elif self.verbose:
                print('[Penguin]: MongoDB connection successful and database is valid.', file=sys.stderr)
        
        except ConnectionFailure as e:
            raise(e)

            
    def _validate_database(self, required_collections):
        """
        Validates the presence of required collections in the connected MongoDB database.

        This method checks if all specified collections in 'required_collections' exist in the 
        currently connected MongoDB database. If the database connection has not been established 
        ('self.db' is None), the validation fails immediately. If any off the collections do not
        exist, they are created and an index is set on either the Scopus or S2 id. If all required 
        collections are present, it returns True.

        Parameters:
        -----------
        required_collections: [str]
            A list of collection names that are required to be present in the database.

        Returns:
        --------
        bool
            True if all required collections are present in the database, False otherwise.
        """
        if self.db is None:
            return False
        
        collection_name = 'your_collection_name'
        existing_collections = self.db.list_collection_names()
        for col_name in required_collections:
            if col_name not in existing_collections:  # collection does not yet exist
                collection = self.db[col_name]
                
                # create an index on either the scopus or s2 id
                if col_name == 'S2':
                    index_field = self.s2_attributes['id']
                elif col_name == 'Scopus':
                    index_field = self.scopus_attributes['id']
                else:
                    warnings.warn(f'Unknown collection name: {col_name!r}')
                    return False
                collection.create_index([(index_field, ASCENDING)], unique=True)
         
        # validated collection name, added index
        return True
            
        
    # 2. Adding documents to the Penguin Database
            
        
    def add_many_documents(self, directory, source, overwrite=True, n_jobs=None):
        """
        Processes a directory of document files that need to be added to the database. The
        function can add both Scopus and S2 documents depending on the `source` agument. 
        Documents will be added in parallel 
        
        Parameters:
        -----------
        directory: str, pathlib.Path
            The directory containing files to be added
        source: str
            The data source (either 'Scopus' or 'S2')
        overwrite: bool, (optional)
            If True and paper id already exists in the collection then the associated data 
            will be updated/overwritten by the new data. Otherwise, this paper id is skipped.
            If paper id does not already exist in collection, this flag has no effect.
            Default is True.
        n_jobs: int, (optional) 
            The number of jobs to run in parallel. If None, the class default for n_jobs will
            be used. Default is None.
        """
        if n_jobs is not None:
            self.n_jobs = n_jobs
        
        if self.n_jobs == 1:
            for file_path in  tqdm(self._file_path_generator(directory, ['.json', '.xml']), disable = not self.verbose):
                self.add_single_document(file_path, source, overwrite=overwrite) 
        else:
            Parallel(n_jobs=self.n_jobs, backend='threading', verbose=self.verbose)(
                delayed(self.add_single_document)(file_path, source, overwrite=overwrite) 
                for file_path in self._file_path_generator(directory, ['.json', '.xml']))

        
    def add_single_document(self, file_path, source, overwrite=True):
        """
        Processes a single data file and adds / updates its content in the database.
        
        This function handles the addition of a Scopus or S2 data file. If the file 
        is from Scopus it will be added to the Scopus collection. If S2, it will be
        added to the S2 collection. In the case of Scopus, the function can hande 
        JSON input (as expected from the Scopus API or iPenguin.Scopus) or XML input
        (the format used by the purchased data).

        Parameters:
        -----------
        file_path: str
            The path to the data file to be processed.
        source: str
            The data source (either 'Scopus' or 'S2')
        overwrite: bool, (optional)
            If True and paper id already exists in the collection then the associated data 
            will be updated/overwritten by the new data. Otherwise, this paper id is skipped.
            If paper id does not already exist in collection, this flag has no effect.
            Default is True.
            
        Returns:
        --------
        None
            Document is added or updated in the database
            
        Raises:
        -------
        ValueError
            If `source` does not match an expected value
        """
        if self.S2_COL.lower() == source.lower():
            self._add_single_s2_file(file_path, overwrite=overwrite)
        elif self.SCOPUS_COL.lower() == source.lower():
            self._add_single_scopus_file(file_path, overwrite=overwrite)
        else:
            raise ValueError(f'Unknown `source` {source!r}')
    
    
    def _file_path_generator(self, directory, ext):
        """
        Generator function that yields paths to files with certain extensions in 
        the given directory.

        Parameters:
        -----------
        directory: str, pathlib.Path
            The directory to search for files
        ext: str, list
            The extension(s) for which to search in the directory. If `ext` is a string
            then the function assumes that a single extension is being sought. If `ext` 
            is a list then it will check for multiple extensions. The file extension(s)
            should be provided in the format of '.EXT' (ex: '.json')

        Yields:
        -------
        pathlib.Path
            Path object representing a file.
        """
        dir_path = pathlib.Path(directory)
        if isinstance(ext, str):
            ext = [ext]
        ext_set = set(ext)
        
        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix in ext_set:
                yield file_path
            
            
    def _add_single_s2_file(self, file_path, overwrite=True):
        """
        Processes a single S2 data file for addition 

        Parameters:
        -----------
        file_path: str
            The path to the data file to be processed.
        overwrite: bool, (optional)
            If True and paper id already exists in the collection then the associated data 
            will be updated/overwritten by the new data. Otherwise, this paper id is skipped.
            Default is True.
            
        Returns:
        --------
        None
            Document is added or updated in the database
            
        Raises:
        -------
        ValueError
            If path is invalid
        """
        if not isinstance(file_path, pathlib.Path):
            file_path = pathlib.Path(file_path)
        
        data = None
        if file_path.suffix == '.json':
            data = process_s2_json(file_path, output_dir=None)
        else:
            raise ValueError(f'S2 `file_path` has an unsupported file extension: {source!r}')
        
        collection = self.db[self.S2_COL]
        id_attr = self.s2_attributes['id']
        if data:
            existing_entry = collection.find_one({id_attr: data[id_attr]})
            if not existing_entry:  # if not exists in collection, add to collection
                collection.insert_one(data)
            elif overwrite:  # if existing_entry and overwrite, update data for id
                collection.update_one({id_attr: data[id_attr]}, {"$set": data})
            else:  # do nothing if existing_entry and no overwrite
                pass
                
                
    def _add_single_scopus_file(self, file_path, overwrite=True):
        """
        Processes a single Scopus data file for addition 

        Parameters:
        -----------
        file_path: str
            The path to the data file to be processed.
        overwrite: bool, (optional)
            If True and paper id already exists in the collection then the associated data 
            will be updated/overwritten by the new data. Otherwise, this paper id is skipped.
            Default is True.
            
        Returns:
        --------
        None
            Document is added or updated in the database
            
        Raises:
        -------
        ValueError
            If path is invalid
        """
        if not isinstance(file_path, pathlib.Path):
            file_path = pathlib.Path(file_path)
        
        data = None
        if file_path.suffix == '.xml':
            data = process_scopus_xml(file_path, output_dir=None)
        elif file_path.suffix == '.json':
            data = process_scopus_json(file_path, output_dir=None)
        else:
            raise ValueError(f'Scopus `file_path` has an unsupported file extension: {source!r}')
        
        collection = self.db[self.SCOPUS_COL]
        id_attr = self.scopus_attributes['id']
        if data:
            existing_entry = collection.find_one({id_attr: data[id_attr]})
            if not existing_entry:  # if not exists in collection, add to collection
                collection.insert_one(data)
            elif overwrite:  # if existing_entry and overwrite, update data for id 
                collection.update_one({id_attr: data[id_attr]}, {"$set": data})
            else:  # do nothing if existing_entry and no overwrite
                pass
    
    
    # 3. Search the Penguin Database

    
    def count_documents(self):
        """
        Return the number of documents in the Scopus and S2.
    
        Returns:
        --------
        dict:
            A dictionary with keys 'scopus' and 's2' containing the number of documents
            in each respective collection.
    
        Raises:
        -------
        Exception
            If there is an error accessing the database collections.
        """
        try:
            scopus_count = self.db[self.SCOPUS_COL].count_documents({})
            s2_count = self.db[self.S2_COL].count_documents({})
            return {
                'scopus': scopus_count,
                's2': s2_count,
            }
        except Exception as e:
            raise Exception(f"Error accessing the database collections: {e}")
            
        
    def text_search(self, target, join='OR', scopus=True, s2=True, as_pandas=True, n_jobs=None):
        """
        Search the Penguin database by text matching in either Scopus documents, S2 documents or both.

        This method allows for searching text across specified fields in the Scopus and S2 collections.
        The `scopus` and `s2` parameters specify which text fields within the documents should be used.
        By default these fields are the title and abstract attributes for each document. The search objective 
        is defined by the `target` parameter. This can be a single string or a list of strings to be found 
        in the text. If this is a list of strings then the relationship between them can be defined using 
        the `join` parameter. This determines whether the results should be joined with a logical 'AND' or 
        'OR'. When searching for text, a document will be returned if the target string is seen in one or 
        more of the text fields being searched. This means that this function supports substring matching 
        and is case insensitive. The results can be returned as a Pandas DataFrame if 'as_pandas' is True, 
        or as MongoDB cursors for each collection if False.

        Parameters:
        -----------
        target: str, [str]
            The text or list of text terms to search for within the specified fields.
        join : str, optional
            The logical join to use across target terms. Can be 'OR' or 'AND'. Default is 'OR'.
        scopus: Iterable, bool, (optional)
            The fields to search within the first document collection. Expected to be an iterable of valid text
            fields found in Scopus. For simplicity, can also be a bool. If True, uses default text attributes 
            defined by Penguin. If False or an empty list, the collection is not searched. Default is True. 
        s2: Iterable, bool, (optional)
            Same behavior as `scopus` but for the S2 documents.
        as_pandas: bool, (optional)
            If True, returns the search results as a SLIC DataFrame. If False, returns a dictionary
            of MongoDB cursors. Default is True.
        n_jobs: int, (optional)
            The number of parallel jobs to use for the query. If specified, overrides the instance's 
            default n_jobs setting.

        Returns:
        --------
        pandas.DataFrame or dict
            If 'as_pandas' is True, returns a Pandas DataFrame containing the combined search results
            from both collections, if applicable. If False, returns a dictionary with keys 'scopus' and 's2'
            containing the cursors to the search results from the respective collections.

        Raises:
        -------
        ValueError
            If the arguments passed to this function are invalid

        Notes:
        ------
        - The search is case-insensitive.
        - The method can search across multiple fields and terms. Fields are always joined by OR. You cannot
          perform a search where a target term is required to be in both the title AND abstract, for example.
          The relationship between different target terms, however, can be specified by the operator used for 
          the `join` parameter.
        - When 'as_pandas' is False, the method returns cursors, which can be iterated over to access 
          the individual documents. These cursors do not store the data but can be thought of as pointers
          to where the data resides on the server (in the database). As a result they are time sensitive
          and become stale if not dereferenced within 10 minutes. They are also useless if conneciton to
          the server is lost.
        """
        if n_jobs is not None:
            self.n_jobs = n_jobs
        
        if not isinstance(join, str) or join.lower() not in {'or', 'and'}:
            raise ValueError('`join` must be in ["or", "and"]')
        if not scopus and not s2:
            raise ValueError('Atleast one of the document collections must be chosen for search')
            
        # setup scopus query
        if isinstance(scopus, bool):
            scopus_query = self.scopus_attributes['text'] if scopus else None
        else:
            scopus_query = self.scopus_attributes['text']
            
        # setup s2 query
        if isinstance(s2, bool):
            s2_query = self.s2_attributes['text'] if s2 else None
        else:
            self.s2_attributes = {'text': s2}
            s2_query = self.s2_attributes['text']
        
        if isinstance(target, str):
            target = [target]
        
        s2_query = create_text_query(s2_query, target, join) if s2 else None
        scopus_query = create_text_query(scopus_query, target, join) if scopus else None
        
        
        if as_pandas:
            return self._query_to_pandas(scopus_query, s2_query)
        else:
            out = {}
            out['s2'] = self.db[self.S2_COL].find(s2_query) if s2 else None
            out['scopus'] = self.db[self.SCOPUS_COL].find(scopus_query) if scopus else None
            return out

    
    def id_search(self, ids, as_pandas=True, n_jobs=None):
        """
        Search the Penguin database by document IDs in either Scopus documents, S2 documents, or both.
    
        This method allows for searching documents across specified collections based on their IDs.
        Each id in `ids` needs to be prefixed with the type of id being evaluted. Current supported 
        prefixes are ['eid', 's2id', 'doi']. This corresponds to Scopus ids, SemanticScholar ids, and 
        DOIs, respectively. The results can be returned as a Pandas DataFrame 
        if 'as_pandas' is True, or as MongoDB cursors for each collection if False.
    
        Parameters:
        -----------
        ids: str, [str]
            The ID or list of IDs to search for within the specified collections.
        as_pandas: bool, optional
            If True, returns the search results as a Pandas DataFrame. If False, returns a dictionary
            of MongoDB cursors. Default is True.
        n_jobs: int, optional
            The number of parallel jobs to use for the query. If specified, overrides the instance's 
            default n_jobs setting.
    
        Returns:
        --------
        pandas.DataFrame or dict
            If 'as_pandas' is True, returns a Pandas DataFrame containing the combined search results
            from both collections, if applicable. If False, returns a dictionary with keys 'scopus' and 's2'
            containing the cursors to the search results from the respective collections.
    
        Raises:
        -------
        ValueError
            If the arguments passed to this function are invalid
    
        Examples:
        ---------
        >>> penguin.id_search(
                ids = [
                    'doi:[doi here]',
                    'doi:[doi here]',
                    's2id:[s2id here]',
                    'eid:[eid here]', 
                    'eid:[eid here]', 
                ]
                as_pandas=True)
        """
        if n_jobs is not None:
            self.n_jobs = n_jobs
        if isinstance(ids, str):
            ids = [ids]
     
        processed_ids = {'eid': [], 's2id': [], 'doi': []}
        for pid in ids:
            prefix, _, actual_id = pid.partition(':')
            if prefix not in {'eid', 's2id', 'doi'}:
                raise ValueError(f"Unknown prefix {prefix:!r} for ID {pid!r}. Valid prefixes are in ['scopus', 's2', 'doi']")
            processed_ids[prefix].append(actual_id)

        # from query for looking up papers by s2id
        s2_query = {}
        if processed_ids['s2id']:
            s2id = self.s2_attributes['id']
            s2_query[s2id] = {"$in": processed_ids['s2id']}

        # form query for looking up papers by scopus id
        scopus_query = {}
        if processed_ids['eid']:
            scopusId = self.scopus_attributes['id']
            scopus_query[scopusId] = {"$in": processed_ids['eid']}

        # form query for looking up papers by doi
        if processed_ids['doi']:
            s2id = self.s2_attributes['doi']
            s2_query[s2id] = {"$in": [x.lower() for x in processed_ids['doi']]}

            scopusId = self.scopus_attributes['doi']
            scopus_query[scopusId] = {"$in": [x.lower() for x in processed_ids['doi']]}

        # transform queries for mongo use
        s2_query = transform_dict(s2_query)
        scopus_query = transform_dict(scopus_query)
        
        # if doi search and id search is involved, add or operator
        s2_query = {'$or': s2_query} if len(s2_query) > 1 else s2_query
        s2_query = None if not s2_query else s2_query
        scopus_query = {'$or': scopus_query} if len(scopus_query) > 1 else scopus_query
        scopus_query = None if not scopus_query else scopus_query
        
        if as_pandas:
            return self._query_to_pandas(scopus_query, s2_query)
        else:
            out = {}
            out['scopus'] = self.db[self.SCOPUS_COL].find(scopus_query) if scopus_query else None
            out['s2'] = self.db[self.S2_COL].find(s2_query) if s2_query else None
            return out

    
    def citation_search(self, target, scopus=False, s2=True, as_pandas=True, n_jobs=None):
        """
        Searches for documents based on citation or reference targets within a specified collection.

        This method allows for searching citations or references within S2 documents. It accepts a target 
        paper ID or a list of target paper IDs targets and retrieves documents citing/referencing these targets. 
        The method can return results as a Pandas DataFrame or a dictionary of cursors, depending on the 
        `as_pandas` flag. The `scopus` parameter is currently not supported and will trigger a warning
        if set to True.

        Parameters:
        -----------
        target: str, [str]
            The document paper IDs for which to look up citations/references
        scopus: bool, (optional)
            Currently not supported. Using this parameter will trigger a warning. This argument is added
            to the function to maintain a similar design structure as the other search functions. Default is False.
        s2: bool, str, (optional)
            Determines the behavior for searching within the S2 collection. If True, uses the default
            citation attribute. If a string is provided, it is used as the citation attribute. Since citations
            and refences use the same data structure, passing the argument 'references' for this parameter
            will trigger a reference search. Default is True. 
        as_pandas: bool, (optional)
            If True, returns the search results as a SLIC DataFrame. If False, returns a dictionary
            of MongoDB cursors. Default is True.
        n_jobs: int, (optional)
            The number of parallel jobs to use for the query. If specified, overrides the instance's 
            default n_jobs setting.

        Returns:
        --------
        pandas.DataFrame or dict
            If 'as_pandas' is True, returns a Pandas DataFrame containing the combined search results
            from both collections, if applicable. If False, returns a dictionary with keys 'scopus' and 's2'
            containing the cursors to the search results from the respective collections.

        Raises:
        -------
        ValueError
            If `s2` is set to False, indicating that no valid collection is selected for the search.
        """
        if n_jobs is not None:
            self.n_jobs = n_jobs
        
        # setup scopus query
        if scopus:
            warnings.warn('[Penguin]: citation_search() does not support Scopus!')
            
        # setup s2 query
        if isinstance(s2, bool):
            s2_query = self.s2_attributes['citations'] if s2 else None
        else:
            self.s2_attributes = {'citations': s2}
            s2_query = self.s2_attributes['citations']
        
        if not s2:
            raise ValueError('S2 must be used for citation/reference search!')
        
        # make sure target is a list
        if not isinstance(target, (list, set, tuple)):
            target = [target]
        else:
            target = list(target)
            
        s2id = self.s2_attributes['id']
        collection = self.db[self.S2_COL]
        target_documents = collection.find({s2id: {"$in": target}})
        if target_documents is None:
            s2_query = None
        else:
            ids = {item[s2id] for doc in target_documents for item in doc.get(s2_query, []) if s2id in item}
            s2_query = {s2id: {"$in": list(ids)}}
            
            
        if as_pandas:
            return self._query_to_pandas(None, s2_query)
        else:
            out = {}
            out['s2'] = self.db[self.S2_COL].find(s2_query) if s2 else None
            out['scopus'] = None
            return out

                        
    def query_by_author(self, collection_name, doc_id, id_attribute='paperId', 
                        author_attribute_list='authors', author_attribute='authorId'):
        """
        Query a MongoDB collection for a document by a direct match and then locate all of the other papers
        from the authors of the target paper that are present in the DB. 

        Parameters:
        -----------
        collection_name: str
            The name of the MongoDB collection to query.
        doc_id: str
            The document ID which to look up.
        id_attribute: str
            The name of the attribute where the document ID is stored. Defaults to 'paperId' for S2 papers.
        author_attribute: str
            The name of the attribute in the documents that contains author information

        Returns:
        --------
        list:
            A list of citing papers
        """
        # query to find a document with given id
        collection = self.db[collection_name]
        target_document = collection.find_one({id_attribute: doc_id})
        if target_document is None:
            return []

        # extract the list of citing papers
        author_ids = [item[author_attribute] for item in 
                      target_document.get(author_attribute_list, []) if author_attribute in item]
        return collection.find({f"{author_attribute_list}.{author_attribute}": {"$in": author_ids}})
    
    
    # 4. Tagging 
    
    
    def resolve_document_id(self, pid):
        """
        Resolves the collection and unique identifier (uid) for a given document_id.
        The document id can either be a Scopus id or a SemanticScholar id.
        
        Parameters:
        -----------
        document_id: str
            The identifier for the document that needs to be resolved. This should be a
            document id (either S2 or Scopus). The id should be prepended by either 'eid:' to
            signify a Scopus document or 's2id:' to signify an S2 document.
        
        Returns:
        --------
        tuple
            A tuple containing the unique identifier (uid) and the associated collection.
        """
        prefix, _, actual_id = pid.partition(':')
        if not actual_id:
            raise ValueError(f'Encountered unknown `id`: {doc_id}. Prepend id with eid: or s2id: to specify id type')
    
        document = None
        collection = None
        if prefix == 'eid':
            collection = self.db[self.SCOPUS_COL]
            secondaryId = self.scopus_attributes['id']
            document = collection.find_one({secondaryId: actual_id})            
        elif prefix == 's2id':
            collection = self.db[self.S2_COL]
            secondaryId = self.s2_attributes['id']
            document = collection.find_one({secondaryId: actual_id})
        else:
            raise ValueError(f'Encountered unknown document id prefix: {prefix}. Valid options are in ["eid", "s2id"]')
    
        if document:
            uid = document['_id']
            return uid, collection
        else:
            return None, None
    
    
    def add_tag(self, document_id, tag):
        """
        Adds a tag to the specified document.
        
        Parameters:
        -----------
        document_id: str
            The identifier for the document to which the tag will be added. This should be a
            document id (either S2 or Scopus). The id should be prepended by either 'eid:' to
            signify a Scopus document or 's2id:' to signify an S2 document.
        tag: str
            The tag to be added to the document.
        
        Returns:
        --------
            None
            
        Example:
        --------
        >>> penguin.add_tag('eid:[eid here]', 'tensor')
        >>> penguin.add_tag('s2id:[s2id here]', 'tensor')
        """
        uid, collection = self.resolve_document_id(document_id)
        if uid and collection is not None:
            collection.update_one(
                {"_id": uid},
                {"$addToSet": {"tags": tag}}
            )
        else:
            raise KeyError(f'Document with id {document_id!r} not found in DB.')
    
    
    def remove_tag(self, document_id, tag):
        """
        Removes a tag from the specified document.
        
        Parameters:
        -----------
        document_id: str
            The identifier for the document to which the tag will be removed. This should be a
            document id (either S2 or Scopus). The id should be prepended by either 'eid:' to
            signify a Scopus document or 's2id:' to signify an S2 document.
        tag: str
            The tag to be removed from the document.
        
        Returns:
        --------
            None
            
        Example:
        --------
        >>> penguin.remove_tag('eid:[eid here]', 'tensor')
        >>> penguin.remove_tag('s2id:[s2id here]', 'tensor')
        """
        uid, collection = self.resolve_document_id(document_id)
        if uid and collection is not None:
            collection.update_one(
                {"_id": uid},
                {"$pull": {"tags": tag}}
            )
        else:
            raise KeyError(f'Document with id {document_id!r} not found in DB.')
        

    def update_tags(self, document_id, new_tags):
        """
        Updates the tags for the specified document with a new set of tags. Used for 
        updating the entire list of tags at once
        
        Parameters:
        -----------
        document_id: str
            The identifier for the document to which the tags will be modified. This should be a
            document id (either S2 or Scopus). The id should be prepended by either 'eid:' to
            signify a Scopus document or 's2id:' to signify an S2 document.
        new_tags: list
            The new set of tags to be assigned to the document.
        
        Returns:
        --------
            None
            
        Example:
        --------
        >>> penguin.update_tags('eid:[eid here]', ['tensor', 'PDE'])
        >>> penguin.update_tags('s2id:[s2id here]', ['tensor', 'PDE'])
        """
        uid, collection = self.resolve_document_id(document_id)
        if uid and collection is not None:
            collection.update_one(
                {"_id": uid},
                {"$set": {"tags": new_tags}}
            )
        else:
            raise KeyError(f'Document with id {document_id!r} not found in DB.')

            
    def find_by_tag(self, tag, as_pandas=True):
        """
        Finds and returns documents that have the specified tag.
        
        Parameters:
        -----------
        tag: str
            The tag to filter documents by.
        as_pandas: bool, (optional)
            If True, returns the search results as a SLIC DataFrame. If False, returns a dictionary
        
        Returns:
        --------
        pandas.DataFrame or dict
            If 'as_pandas' is True, returns a Pandas DataFrame containing the combined search results
            from for the tag from Scopus and S2 collections. If False, returns a dictionary with keys 
            'scopus' and 's2' containing the cursors to the search results from the respective collections.
        """
        if as_pandas:
            query = {"tags": tag}
            return self._query_to_pandas(query, query)
        else:
            out = {}
            out['scopus'] = self.db[self.SCOPUS_COL].find({"tags": tag}) 
            out['s2'] = self.db[self.S2_COL].find({"tags": tag})
            return out

    
    # 5. iPenguin Hook
    
    
    def get_id_bloom(self, source, max_items=1.25, false_positive_rate=0.001):  
        """
        Initializes a Bloom filter with IDs from a specified source collection.

        This method selects a collection based on the 'source' parameter, which should match
        one of the predefined S2 or Scopus collection names (S2_COL or SCOPUS_COL). It then retrieves 
        IDs from the selected collection and adds them to a Bloom filter. The Bloom filter is configured 
        based on the estimated number of items ('max_items') and the desired false positive rate.

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

        Notes:
        ------
        - The Bloom filter is a probabilistic data structure that is used to test whether an element
          is a member of a set. False positive matches are possible, but false negatives are not.
        - The 'max_items' parameter impacts the size and the false positive rate of the Bloom filter.
          Adjusting this parameter can optimize the performance and accuracy based on the expected
          dataset size.
        - The function retrieves only the ID attribute from the documents in the collection, 
          excluding the MongoDB '_id' field, to populate the Bloom filter.
        """
        id_attr = None
        collection = None
        if self.S2_COL.lower() == source.lower():
            collection = self.db[self.S2_COL]
            id_attr = self.s2_attributes['id']
        elif self.SCOPUS_COL.lower() == source.lower():
            collection = self.db[self.SCOPUS_COL]
            id_attr = self.scopus_attributes['id']
        else:
            raise ValueError(f'Unknown `source` {source!r}')
        
        num_documents = collection.count_documents({})
        if isinstance(max_items, int):
            num_documents = max_items
        elif isinstance(max_items, float):
            num_documents *= max_items
            num_documents = int(num_documents)
        else:
            raise ValueError('`max_items` is expected to be a float or an int')
        
        bf = Bloom(num_documents, false_positive_rate)  # initialize a bloom filter
        cursor = collection.find({}, {id_attr: 1, '_id': 0})  # retrieve only eid/s2id attribute and exclude '_id' field
        for doc in cursor:
            if id_attr in doc:
                if not isinstance(doc[id_attr], str):
                    continue
                if id_attr == 'eid':
                    bf.add(doc[id_attr][7:])  # remove the '2-s2.0-' prefix from Scopus ids
                else:
                    bf.add(doc[id_attr])
        return bf
        
    
    def _query_to_pandas(self, scopus_query, s2_query):
        """
        Retrieves data based on two separate queries to Scopus and S2 collections and transforms them 
        into a single Pandas DataFrame in SLIC format.

        This method executes two different queries using the `_query_to_pandas_helper` method for each. 
        The first query (`scopus_query`) is executed against a collection specified by `self.SCOPUS_COL`, 
        and the second query (`s2_query`) is executed against another collection specified by `self.S2_COL`. 
        Each query's results are independently transformed into a DataFrames using their own handler
        functions. The two DataFrames are then combined into a single DataFrame by the `form_df` function, 
        which is in the standard SLIC format.

        Parameters:
        -----------
        scopus_query: dict
            The MongoDB query to be executed against the Scopus documents
        s2_query: dict
            The MongoDB query to be executed against the S2 documents

        Returns:
        --------
        pd.DataFrame
            The joint DataFrame for both Scopus and S2. Formatted to be used with the rest of the Zoo libraries. 
        """
        scopus_df = self._query_to_pandas_helper(scopus_query, self.SCOPUS_COL, form_scopus_df)
        s2_df = self._query_to_pandas_helper(s2_query, self.S2_COL, form_s2_df)
        return form_df(scopus_df, s2_df)

    
    def _query_to_pandas_helper(self, query, col, processor):
        """
        Retrieves data from a MongoDB collection based on a query and turns it into a 
        Pandas DataFrame.

        This method first divides the result set of the query into multiple cursors for parallel 
        processing, if applicable. Each cursor represents a subset of the query results. The method 
        then processes these subsets in parallel, converts them into Pandas DataFrame slices, and 
        concatenates these slices into a single DataFrame. If the number of jobs (cursors) is one, 
        it processes the query result without parallelization. If the query is None, it returns an empty 
        DataFrame with the appropriate S2 or Scopus headers. 

        Parameters:
        -----------
        query: dict or None
            The MongoDB query to execute. If None, an empty DataFrame is returned.
        col: str
            The name of the MongoDB collection to query.
        processor: callable
            A function that takes a cursor, an empty list, (or any other iterable) and returns a Pandas DataFrame. 
            This function is responsible for converting the raw data from MongoDB into a DataFrame format. The
            currently available options are `form_scopus_df()` and `form_s2_df()` for the Scopus collection and
            the S2 collection respectively.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the data retrieved from MongoDB based on the provided query, processed by the 'processor' 
            function. If the query is None, returns an empty DataFrame with just the header.
        """
        if query is not None:
            cursors = self._parallel_cursors(self.db[col], query)
            n_jobs = len(cursors)
            if n_jobs > 1:  # generate dataframe slices in parallel
                jobs = Parallel(n_jobs=len(cursors), backend='threading', verbose=self.verbose)(
                    delayed(processor)(c) for c in cursors)
                df = pd.concat(jobs, sort=False)
                return df.reset_index(drop=True)
            else:
                return processor(next(iter(cursors)))  # single core processing
        else:
            return processor([])  # create empty df (just header basically)

    
    def _parallel_cursors(self, collection, query):
        """
        Generates a list of cursors for a MongoDB collection, each pointing to a subset of documents
        based on the specified query. The total number of documents matched by the query is divided
        as evenly as possible among the specified number of jobs (`self.n_jobs`). 

        Parameters:
        -----------
        collection : pymongo.collection.Collection
            The MongoDB collection to query.
        query : dict
            The query to execute on the collection.

        Returns:
        --------
        list
            A list of pymongo.cursor.Cursor objects, each configured with skip and limit to iterate over
            a subset of the query results. The division of documents is as even as possible, with any
            remainder from the division process distributed among the first cursors in the list.

        Note:
        -----
        This function is particularly useful for parallel processing of MongoDB query results. Care should be
        taken when using a large number of cursors, as each cursor consumes server resources.
        """
        total_docs = collection.count_documents(query)
        n_jobs = min(self.n_jobs, total_docs) if total_docs > 0 else 1
        chunk_size, extras = divmod(total_docs, n_jobs)

        cursors = []
        skip_amount = 0
        for i in range(n_jobs):
            limit_amount = chunk_size + (1 if i < extras else 0)
            cursor = collection.find(query).skip(skip_amount).limit(limit_amount)
            cursors.append(cursor)
            skip_amount += limit_amount
        return cursors
    
    
    # Setters / Getters
    
    
    @property
    def uri(self):
        return self._uri

    @uri.setter
    def uri(self, value):
        if not isinstance(value, str):
            raise TypeError('`uri` must be a string!')
        if not re.match(r'^[a-zA-Z0-9.-]+:[0-9]+$', value):
            raise ValueError("`uri` must be in the format 'hostname:port'")
        self._uri = value
    
    @property
    def db_name(self):
        return self._db_name

    @db_name.setter
    def db_name(self, value):
        if not isinstance(value, str):
            raise ValueError('`db_name` must be a string!')
        self._db_name = value
    
    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, value):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("`username` must be a non-empty string or None")
        self._username = value

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, value):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("`password` must be a non-empty string or None")
        self._password = value
    
    @property
    def n_jobs(self):
        return self._n_jobs
    
    @n_jobs.setter
    def n_jobs(self, n_jobs):
        self._n_jobs = verify_n_jobs(n_jobs)
        
    @property
    def scopus_attributes(self):
        return self._scopus_attributes
    
    @scopus_attributes.setter
    def scopus_attributes(self, scopus_attributes):
        self._scopus_attributes = verify_attributes(scopus_attributes, self.scopus_attributes, 'scopus_attributes')
        
    @property
    def s2_attributes(self):
        return self._s2_attributes
    
    @s2_attributes.setter
    def s2_attributes(self, s2_attributes):
        self._s2_attributes = verify_attributes(s2_attributes, self.s2_attributes, 's2_attributes')
        
    @property
    def verbose(self):
        return self._verbose
        
    @verbose.setter
    def verbose(self, verbose):
        if isinstance(verbose, bool):
            self._verbose = int(verbose)  # convert False to 0, True to 1
        elif isinstance(verbose, int):
            if verbose < 0:
                raise ValueError('Integer values for `verbose` must be non-negative!')
            self._verbose = verbose
        else:
            raise TypeError('`verbose` should be of type bool or int!')