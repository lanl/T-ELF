#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© 2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
"""

try:
    from mpi4py import MPI
except:
    MPI = None

from .pre_process import simple_document_clean
from .pre_process import check_character_lengths
from .pre_process import _organize_simple_clean_defaults
from .pre_process import advance_document_clean
from .pre_process import lemmatize_document
from .pre_process import stem_document
from .pre_process import build_vocab_stem_subs
from .pre_process import correct_text

from .default_stop_words import default_stop_words
from .default_stop_phrases import default_stop_phrases
from .detect_nonenglish import is_english

from joblib import Parallel, delayed
from tqdm import tqdm
import  multiprocessing, sys, pickle, time, spacy, os, ast, \
        pandas as pd, numpy as np
 


class Vulture():

    def __init__(self,
                 n_jobs=-1,
                 n_nodes=1,
                 verbose=10,
                 min_characters=2,
                 min_unique_characters=2,
                 clean_text=True,
                 advance_clean=False,
                 allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', "PROPN"],
                 spacy_model="en_core_web_lg",
                 lemmatize=True,
                 lemmatize_spacy=False,
                 lemmatize_slic=False,
                 stem=False,
                 remove_nonenglish=True,
                 remove_nonenglish_params=(0.9, 0.175),
                 parallel_backend="multiprocessing",
                 simple_clean_settings={}
                 ):
        """
        Vulture is a parallel, multi-node parallel, and distributed parallel document
        pre-processing tool.
        It is designed to be simple and fast.

        Vultures are natures' cleaners!

        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel processes. The default is -1.
        n_nodes : int, optional
            Number of nodes. The default is 1.
        verbose : int, optional
            Verbosity level. The default is 10.
        min_characters : int, optional
            Minimum number of characters a token should have. The default is 2.
        min_unique_characters : int, optional
            Minimum number of unique characters a token should have.. The default is 2.
        clean_text : bool, optional
            If True, perform cleaning. The default is True.
        advance_clean : bool, optional
            If True, and clean_text is also True, then performs advance cleaning. The default is False.
        allowed_postags : list, optional
            List of allowed postags for Spacy. The default is ['NOUN', 'ADJ', 'VERB', 'ADV', "PROPN"].
        spacy_model : string, optional
            Spacy model to load. The default is "en_core_web_lg".
        lemmatize : bool, optional
            If True, performs lemmatization. Uses WordNet lemmatization. The default is True.
        lemmatize_spacy : bool, optional
            If True, performs lemmatization on the Spacy model during advance pre-processing. 
            advance_clean must be true to use this.
            The default is False.
        lemmatize_slic : bool, optional
            If True, performs lemmatization on the nltk model during advance pre-processing. 
            advance_clean must be true to use this.
            The default is Fasle.
        stem : bool, optional
            If True, performs stemming. The default is True.
        remove_nonenglish : bool, optional
            Use heuristic to remove foreign language papers
        remove_nonenglish_params : (float, float), optional
            A tuple of floats that control the heuristic. The first entry in the tuple is the minimum 
            acceptable ratio for ascii characters to total characters in the text. The second float 
            is the min acceptable ratio of stopwords in the text
        simple_clean_settings: dict, optional
            Settings used in simple cleaning. See `pre_process._organize_simple_clean_defaults <https://github.com/lanl/T-ELF/blob/main/TELF/pre_processing/Vulture/pre_process.py#L356>`_ for defaults.
            
            .. note::

                **Example Settings:**

                .. code-block:: python

                    simple_clean_settings= { 
                        "remove_copyright_with_symbol": True,
                        "remove_stop_phrases": False,
                        "make_lower_case": True,
                        "remove_trailing_dash": True,  # -foo-bar- -> foo-bar
                        "make_hyphens_words": False,  # foo-bar -> foobar
                        "remove_next_line": True,
                        "remove_email": True,
                        "remove_dash": False,
                        "remove_between_[]": True,
                        "remove_between_()": True,
                        "remove_[]": False,
                        "remove_()": False,
                        "remove_\\": False,
                        "remove_^": False,
                        "remove_numbers": False,
                        "remove_nonASCII": False,
                        "remove_tags": True,  # remove HTML tags
                        "remove_special_characters": True,  # option to specify these?
                        "remove_stop_words": True,
                    }


        Returns
        -------
        None.

        """

        # input check
        assert type(n_jobs) == int, "n_jobs must be a type of integer!"
        assert n_nodes >= 1, "n_nodes must be 1 or greater value!"
        assert clean_text or lemmatize or stem, \
            "At least one must be True for clean_text, lemmatize, stem!"
        if advance_clean and not clean_text:
            sys.exit(
                "advance_clean was True but clean_text was False." +
                "To use advance cleaning, set both clean_text and advance_clean to True!")

        if lemmatize_spacy and not advance_clean:
            sys.exit(
                "lemmatize_spacy was True but advance_clean was False." +
                "To use lemmatization with Spacy, set both lemmatize_spacy and advance_clean to True!" +
                "If only want to do lemmatization but not advance cleaning, use lemmatize=True instead.")
        
        ### Commented out to allow lemmatize_slic to work without using advance_clean
        #if lemmatize_slic and not advance_clean:
        #    sys.exit(
        #        "lemmatize_slic was True but advance_clean was False." +
        #        "To use lemmatization with nltk, set both lemmatize_slic and advance_clean to True!" +
        #        "If only want to do lemmatization but not advance cleaning, use lemmatize=True instead.")

        if n_nodes > 1 and MPI is None:
            sys.exit("Attempted to use n_nodes>1 but MPI is not available!")

        # if allowed postags is list make it a hashmap
        if isinstance(allowed_postags, list) or isinstance(allowed_postags, np.ndarray):
            allowed_postags = dict(zip(allowed_postags, [1]*len(allowed_postags)))

        ascii_ratio, stopwords_used = remove_nonenglish_params
        assert ascii_ratio > 0 and ascii_ratio < 1, 'ASCII ratio must be on [0, 1]'
        assert stopwords_used > 0 and stopwords_used < 100, 'ASCII ratio must be on [0, 100]'
            
        self.n_jobs = n_jobs
        self.n_nodes = n_nodes
        self.verbose = verbose
        self.min_characters = min_characters
        self.min_unique_characters = min_unique_characters
        self.clean_text = clean_text
        self.advance_clean = advance_clean
        self.allowed_postags = allowed_postags
        self.spacy_model = spacy_model
        self.lemmatize = lemmatize
        self.lemmatize_spacy = lemmatize_spacy
        self.lemmatize_slic = lemmatize_slic
        self.stem = stem
        self.remove_nonenglish = remove_nonenglish
        self.remove_nonenglish_ar = ascii_ratio
        self.remove_nonenglish_su = stopwords_used
        self.parallel_backend = parallel_backend
        self.simple_clean_settings = _organize_simple_clean_defaults(simple_clean_settings)

    def clean(self,
              documents: dict,
              stop_words={},
              stop_phrases=[],
              filename="cleaned_documents",
              substitutions: dict={},
             ) -> None:
        """
        Performs single-node parallel pre-processing, or multi-node parallel pre-processing.
        Entire corpus is given at once, and each node will work on certain segment of the documents
        when using multi-node parallel pre-processing.


        Parameters
        ----------
        documents : dict
            Dictionary of documents to clean. In this dictionary, keys are the unique document
            identifiers, and values are the text to clean.
        stop_words : dict (recommended), or list
            Hashmap (dict) of stopwords. O(1) lookup.
            List of stopwords. O(n) lookup. Default is {}.
        stop_phrases: list, optional
            List of phrases to be removed.
        filename : string, optional
            Pattern and path when saving the results. The default is "cleaned_documents".
        substitutions : dict 
            src_word as keys, destination words as values. Keys should not be in values, and v.v.

        Returns
        -------
        None

        """

        assert type(
            documents) == dict, "Required format: documents= {'id1':'text', 'id2':'text', ...}."

        if self.verbose:
            start_time = time.time()

        # if no stop-words given, use the default
        if len(stop_words) == 0:
            stop_words = default_stop_words.copy()

        # if no stop-phrases given, use the default
        if len(stop_phrases) == 0:
            stop_phrases = default_stop_phrases.copy()

        # if stop words is a list, make it a Hashmap
        if isinstance(stop_words, list) or isinstance(stop_words, np.ndarray):
            stop_words = dict(zip(stop_words, [1]*len(stop_words)))
            
        save_path = self._get_parent_directory(filename)
        if self.lemmatize_slic:
            tokens = self._parallel_helper(documents, {}, self._tokenize_helper, as_list=True)
            before_clean_path = os.path.join(save_path,'before_any_cleaning-vocabulary.txt')
            self._save_tokens( before_clean_path, tokens )
            
        # using more than one node, then get this nodes' chunk
        if self.n_nodes > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            filename = f'{filename}_node-{rank}'
            document_chunks = self._split_dict_chunks(input_dict=documents, chunks=self.n_nodes)
            documents = document_chunks[rank]
        else:
            comm = None
            
            
        if self.remove_nonenglish:
            if self.verbose:
                print("Removing non-english documents through heuristic.")   
            non_english_docs = self._parallel_helper(
                documents, {"ascii_ratio": self.remove_nonenglish_ar, 
                            "stopwords_used": self.remove_nonenglish_su}, self._nonenglish_helper)
            
            documents = {k:v for k,v in documents.items() if k not in non_english_docs}
            if self.verbose:
                print(f"Removed {len(non_english_docs)} non-english documents.")   
            
        # parallel clean
        if self.clean_text:

            # simple clean
            if self.verbose:
                print("Performing simple-preprocess.")

            results = self._parallel_helper(
                documents, {"stop_words": stop_words,
                            "stop_phrases": stop_phrases,
                            "clean_settings": self.simple_clean_settings
                            },
                self._clean_helper)

            # advance clean
            if self.advance_clean:
                if self.verbose:
 
                    if self.lemmatize_spacy:
                        print("Performing advance pre-process with Spacy lemmatization.")
                    else:
                        print("Performing advance pre-process.")

                results = self._parallel_helper(
                    results, {
                        "allowed_postags": self.allowed_postags,
                        "lemmatize": self.lemmatize_spacy
                    }, self._advance_clean_helper)
  
        else:
            results = documents

        # parallel lematization, stemming
        if self.lemmatize:
            if self.verbose:
                print("Performing lemattization.")
            results = self._parallel_helper(results, {}, self._lemmatize_helper)

        if self.stem:
            if self.verbose:
                print("Performing stemming.")
            results = self._parallel_helper(results, {}, self._stem_helper)

        # check the character lengths
        if self.simple_clean_settings["check_char_length"]:
            if self. verbose:
                print("Performing character length checks.")

            results = self._parallel_helper(
                results, {"min_characters": self.min_characters,
                          "min_unique_characters": self.min_unique_characters},
                self._char_length_helper)
        
        if self.lemmatize_slic:
            if self.verbose:
                print("Performing advance pre-process with Slic lemmatization (nltk stems).")
            
            # get all tokens as a quasi_vocabulary
            quasi_vocabulary = list(self._parallel_helper(results, {}, self._tokenize_helper, as_list=True))
            before_slic_lemma = os.path.join(save_path,'before_slic_lemma-vocabulary.txt')
            self._save_tokens( before_slic_lemma, quasi_vocabulary ) 
            
            if self. verbose:
                first_ten_tokens = quasi_vocabulary[:10]
                token_length = len(quasi_vocabulary)
                print(f'Token length: {token_length}, First ten tokens before slic lemma: {first_ten_tokens}')
                
            subs_stemmed = build_vocab_stem_subs(quasi_vocabulary) # suffixes in future, when unification occurs.
            df_stem_map = pd.DataFrame(list(subs_stemmed.items()))
            df_stem_map.to_csv( os.path.join(save_path,'vocabulary_stem_map.csv'), index=False)   

            results = self._parallel_helper(results, {'corrections':subs_stemmed }, correct_text)
        
            # tokenize, save to path
            tokens = self._parallel_helper(results, {}, self._tokenize_helper, as_list=True)
            after_clean_path = os.path.join(save_path,'after_slic_lemma-vocabulary.txt')
            self._save_tokens( after_clean_path, tokens ) 
            
        if substitutions:
            results = self._parallel_helper(results, {'corrections':substitutions }, correct_text)

        # save pre-processing results
        pickle.dump(results, open(filename + ".p", "wb"))

        # show the time
        if self.verbose:
            end_time = time.time()
            print("Done. Time=", str(end_time - start_time))

    def dataframe_clean( self, data: pd.DataFrame, columns: list, append_to_original_df: bool=False, concat_cleaned_cols: bool=False, stop_words: list=[], stop_phrases: list=[], data_path: str=None, overwrite:bool=True) -> pd.DataFrame:
        """
        Wrapper function for Vulture text cleaning. Cleans the text from a chosen DataFrame columms and applies
        cleaning. Then formats the cleaned data and returns as a new DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing text to be cleaned
        columns : list
            The name of the text column that needs to be cleaned
        append_to_original_df: bool, optional
            If the cleaned columns of the dataframe should be appended on the columns of the original dataframe
        concat_cleaned_cols:bool, optional
            If the cleaned columns of the dataframe should be concatenated together into a single common column
        stop_words : list, optional
            Words that will be removed during cleaning
        stop_phrases: list, optional
            Phrases to be removed.
        data_path : str, optional
            Vulture outputs the cleaned text as pickle files. This argument specifies the path
            at which they are saved. These pickle files are later loaded and converted to a DataFrame.
        overwrite : bool, optional
            Flag that controls if results should be overwritten. If true, cleaning will be performed
            regardless of existence of previous data. Default is True.

        Returns
        -------
        pd.DataFrame
            if not concateneated, DataFrame containing the cleaned results
            if concateneated, original DataFrame with appended columns containing the cleaned results
        """
        if not all(col in data.columns for col in columns):
            missing_cols = set(columns) - set(data.columns)
            print(f"Error: the following columns are missing from the dataframe: {missing_cols}")
            return None
        
        df = pd.DataFrame()

        if concat_cleaned_cols :
            concatenated_columns = 'cleaned_' +  '_'.join(columns)
            selected_columns_data = data[columns]
            concated_string_rows = selected_columns_data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            
            if overwrite or not os.path.exists(os.path.join(data_path, f'{concatenated_columns}.p')):
                documents = {idx: text for idx, text in enumerate(concated_string_rows.tolist()) }
                self.clean(documents, stop_words, stop_phrases, filename=os.path.join(data_path, concatenated_columns))
            
                clean_documents = pickle.load(open(os.path.join(data_path, f'{concatenated_columns}.p'), 'rb'))
                df[concatenated_columns] = clean_documents

        else:
            for col in columns:
                clean_col = f'clean_{col}'

                if overwrite or not os.path.exists(os.path.join(data_path, f'{clean_col}.p')):
                    zipped_data = zip(data.index.to_list(), data[col].to_list())
                    documents = {idx: text for idx, text in zipped_data }
                    self.clean(documents, stop_words, stop_phrases, filename=os.path.join(data_path, clean_col))
            
                clean_documents = pickle.load(open(os.path.join(data_path, f'{clean_col}.p'), 'rb'))
                df[clean_col] = clean_documents

        if append_to_original_df:
            common_cols = data.columns.intersection(df.columns)
            new_cols = df.columns.difference(data.columns)

            data = pd.concat([data, df[new_cols]], axis=1)
            data.update(df[common_cols])

            return data

        return df

    def distributed_clean(self,
                          files: list,
                          stop_words={},
                          stop_phrases=[],
                          filename="cleaned_documents",
                          file_loader=None) -> None:
        """
        Each n node loads a segment of the set of files, and performs parallel pre-processing on
        the documents from the loaded files.

        Use this approach when the corpus cannot fit in the memmory at once.

        Parameters
        ----------
        files : list
            List if strings for locations to the files.
        stop_words : dict (recommended), or list
            Hashmap (dict) of stopwords. O(1) lookup.
            List of stopwords. O(n) lookup. Default is {}.
        stop_phrases: list, optional
            List of phrases to be removed.
        filename : string, optional
            Pattern and path when saving the results. The default is "cleaned_documents".
        file_loader : function, optional
            Custom function for loading the files. The default is None.
            It should take in string path to the file, and return dictionary
            where the keys are the unique document IDs and values are the text.

        Returns
        -------
        None

        """

        assert self.n_nodes > 1, "n_nodes must be greater than 1!"
        files_chunks = np.array_split(files, self.n_nodes)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        filename = f'{filename}_node-{rank}'
        node_files = files_chunks[rank]
        if self.verbose:
            start_time = time.time()

        # if no stop-words given, use the default
        if len(stop_words) == 0:
            stop_words = default_stop_words.copy()

        # if no stop-phrases given, use the default
        if len(stop_phrases) == 0:
            stop_phrases = default_stop_phrases.copy()

        # if stop words is a list, make it a Hashmap
        if isinstance(stop_words, list) or isinstance(stop_words, np.ndarray):
            stop_words = dict(zip(stop_words, [1]*len(stop_words)))

        for idx, file in tqdm(enumerate(node_files), total=len(node_files), disable=not self.verbose):

            # name of the file for saving
            save_name = f'{filename}_file-{idx}'

            # use custom function to load the file
            if callable(file_loader) and (file_loader is not None):
                documents = file_loader(file)

            # default load from Pickle
            else:
                documents = pickle.load(open(file, "rb"))

            # check the type of the document
            assert type(
                documents) == dict, "Required format: documents= {'id1':'text', 'id2':'text', ...}."

            # parallel clean
            if self.clean_text:

                # simple clean
                if self.verbose:
                    print("Performing simple-preprocess.")
                results = self._parallel_helper(
                    documents, {"stop_words": stop_words,
                                "stop_phrases": stop_phrases,
                                "clean_settings": self.simple_clean_settings},
                    self._clean_helper)

                # advance clean
                if self.advance_clean:
                    if self.verbose:
                        if self.lemmatize_spacy:
                            print("Performing advance pre-process with Spacy lemmatization.")
                        else:
                            print("Performing advance pre-process.")

                    results = self._parallel_helper(
                        results, {
                            "allowed_postags": self.allowed_postags,
                            "lemmatize": self.lemmatize_spacy
                        }, self._advance_clean_helper)
            else:
                results = documents

            # parallel lematization, stemming
            if self.lemmatize:
                if self.verbose:
                    print("Performing lemattization.")
                results = self._parallel_helper(results, {}, self._lemmatize_helper)

            if self.stem:
                if self.verbose:
                    print("Performing stemming.")
                results = self._parallel_helper(results, {}, self._stem_helper)

            # check the character lengths
            if self.simple_clean_settings["check_char_length"]:

                if self. verbose:
                    print("Performing character length checks.")

                results = self._parallel_helper(
                    results, {"min_characters": self.min_characters,
                              "min_unique_characters": self.min_unique_characters},
                    self._char_length_helper)

            # save pre-processing results
            pickle.dump(results, open(save_name + ".p", "wb"))

        # show the time
        if self.verbose:
            end_time = time.time()
            print("Done. Time=", str(end_time - start_time))

    def _split_dict_chunks(self, input_dict: dict, chunks: int) -> list:
        """
        Splits the given dictionary into list of multiple dictionaries.

        Parameters
        ----------
        input_dict : dict
            Dictionary to split.
        chunks : int
            How many sets of dictionaries to create.

        Returns
        -------
        list
            List of dictionaries.

        """

        return_list = [dict() for idx in range(chunks)]
        idx = 0
        for k, v in input_dict.items():
            return_list[idx][k] = v
            if idx < chunks-1:
                idx += 1
            else:
                idx = 0
        return return_list

    
    
    def _parallel_helper(self, documents: dict, parameters: dict, function, as_list: bool = False) -> dict:
        """
        Helper function to run processing of given documents in parallel

        Parameters
        ----------
        documents : dict
            Dictionary of documents to clean. In this dictionary, keys are the unique document
            identifiers, and values are the text to clean.
        parameters : dict
            Parameters of the function to use for processing, except documents.
        function : callable
            Processing function to call.
        

        Returns
        -------
        dict
            processed docuements, where keys are the document IDs and values are the text.

        """

        # adjust for negative number of job requests
        if self.n_jobs <= 0:
            n_chunks = multiprocessing.cpu_count()
            n_chunks -= (np.abs(self.n_jobs) + 1)
        else:
            n_chunks = self.n_jobs

        # split the documents into chunks
        document_chunks = self._split_dict_chunks(documents, n_chunks)

        # process each chunk of documents
        list_results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend=self.parallel_backend)(
            delayed(function)(documents=chunk, **parameters) for chunk in document_chunks)

        # flatten the results
        results = []
        for job_result in list_results:
            results += job_result
        
        if as_list:
            return results

        return dict(results)

    
    def _nonenglish_helper(self, documents: dict, ascii_ratio: float, stopwords_used: int) -> dict:
        """
        remove non-english documents

        Parameters
        ----------
        documents : dict
            Dictionary of documents to clean. In this dictionary, keys are the unique document
            identifiers, and values are the text to clean.

        Returns
        -------
        list of tuples
            A list of (document id, bool) tuples where the bool is a determination of whether the 
            given document is English or not.
        """
        results = []
        for docID, doc in documents.items():
            doc_is_english = is_english(doc, ascii_ratio, stopwords_used)
            if not doc_is_english:
                results.append((docID, doc_is_english))
        return results

    
    def _stem_helper(self, documents: dict) -> dict:
        """
        stemming of the documents.

        Parameters
        ----------
        documents : dict
            Dictionary of documents to clean. In this dictionary, keys are the unique document
            identifiers, and values are the text to clean.

        Returns
        -------
        dict
            stemmed documents, where keys are the document IDs and values are the text..

        """
        results = []
        for docID, doc in documents.items():
            results.append(stem_document(document=doc, document_id=docID))

        return results

    def _lemmatize_helper(self, documents: dict) -> dict:
        """
        lemmatization of the documents.

        Parameters
        ----------
        documents : dict
            Dictionary of documents to clean. In this dictionary, keys are the unique document
            identifiers, and values are the text to clean.

        Returns
        -------
        dict
            lemmatized documents, where keys are the document IDs and values are the text..

        """
        results = []
        for docID, doc in documents.items():
            results.append(lemmatize_document(document=doc, document_id=docID))

        return results
    

    def _clean_helper(self,
                      documents: dict,
                      stop_words,
                      stop_phrases: list,
                      clean_settings: dict) -> dict:
        """
        cleaning of the documents.

        Parameters
        ----------
        documents : dict
            Dictionary of documents to clean. In this dictionary, keys are the unique document
            identifiers, and values are the text to clean.
        stop_words : dict (recommended), or list
            Hashmap (dict) of stopwords. O(1) lookup.
            List of stopwords. O(n) lookup. Default is {}.
        stop_phrases: list, optional
            List of phrases to be removed.
        clean_settings: dict, optional
            Settings used in simple cleaning. See _organize_simple_clean_defaults for defaults.

        Returns
        -------
        dict
            cleaned documents, where keys are the document IDs and values are the text.

        """
        results = []
        for docID, doc in documents.items():
            results.append(simple_document_clean(
                document=doc,
                document_id=docID,
                stop_words=stop_words,
                stop_phrases=stop_phrases,
                clean_settings=clean_settings))

        return results

    def _char_length_helper(self,
                            documents: dict,
                            min_characters: int,
                            min_unique_characters: int) -> dict:
        """
        checking of character lengths

        Parameters
        ----------
        documents : dict
            Dictionary of documents to clean. In this dictionary, keys are the unique document
            identifiers, and values are the text to clean.
        min_characters : int, optional
            Minimum number of characters a token should have. The default is 2.
        min_unique_characters : int, optional
            Minimum number of unique characters a token should have. The default is 2.
        clean_settings: dict, optional
            Settings used in simple cleaning. See _organize_simple_clean_defaults for defaults.

        Returns
        -------
        dict
            cleaned documents, where keys are the document IDs and values are the text.

        """
        results = []
        for docID, doc in documents.items():
            results.append(check_character_lengths(
                document=doc,
                document_id=docID,
                min_characters=min_characters,
                min_unique_characters=min_unique_characters))

        return results

    def _advance_clean_helper(self,
                              documents: dict,
                              allowed_postags: list,
                              lemmatize: bool) -> dict:
        """
        Advance cleaning of the documents.

        Parameters
        ----------
        documents : dict
            Dictionary of documents to clean. In this dictionary, keys are the unique document
            identifiers, and values are the text to clean.
        allowed_postags : list, optional
            The list of allowed Postags. The default is ['NOUN', 'ADJ', 'VERB', 'ADV', "PROPN"].

        Returns
        -------
        dict
            cleaned documents, where keys are the document IDs and values are the text.

        """
        results = []
        nlp = spacy.load(self.spacy_model)
        for docID, doc in documents.items():
            results.append(advance_document_clean(
                document=doc,
                document_id=docID,
                allowed_postags=allowed_postags,
                nlp=nlp,
                lemmatize=lemmatize
            ))

        return results
    
    def _tokenize_helper(self, documents: dict ) -> list:
        """
        tokenize documents

        Parameters
        ----------
        documents : dict
            Dictionary of documents to clean. In this dictionary, keys are the unique document
            identifiers, and values are the text to clean.
         
        Returns
        -------
        list
            tokens in the documentss

        """
        results = []
        for  doc in documents.values():
            results.extend(doc.split())
            
        return results #list(set(results))
    
    def _get_parent_directory(self, path):
        """
        Returns parent dir unless path is a directory.

        Parameters
        ----------
        path : str
            can be a file or a dir
         
        Returns
        -------
        itself or parent dir
        """
        if os.path.isdir(path):
            return path
        return os.path.dirname(path)
    
    def _save_tokens(self, save_path: str, tokens: list ):
        """
        Saves tokens

        Parameters
        ----------
        save_path : str
            Location to save tokens, includes dir path and file
        tokens : list
            List of tokens in corpus to save

        Returns
        -------
        None
        """
        directories = os.path.dirname(save_path) 
        
        try:
            os.makedirs(directories)
            if self.verbose:
                print(f"{directories} doesn't exist, creating...")
                
        except (FileExistsError, FileNotFoundError):
            if self.verbose:
                print(f"Directories '{directories}' already exists.")

        # save tokens
        with open(save_path, 'w') as token_file:
            for item in tokens:
                token_file.write(f"{item}\n")  
