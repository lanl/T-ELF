#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:38:26 2021

@author: maksimekineren
"""
try:
    from mpi4py import MPI
except:
    MPI = None
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import sparse
from joblib import Parallel, delayed
import multiprocessing
import sys

def _chunk_list(l, n):
    """
    Yield n number of striped chunks from l.

    Parameters
    ----------
    l : list
        list to be chunked.
    n : int
        number of chunks.

    Yields
    ------
    list
        chunks.

    """
    for i in range(0, n):
        yield l[i::n]

def _co_occurance_parallel_helper(documents, verbose, window_size, sentences, V_map):
    """
    Parallel helper for counting word occurances

    Parameters
    ----------
    documents: list
        List of documents. Each entry in the list contains text.
        if sentences=True, then documents is a list of lists, where
        each entry in documents is a list of sentences.
    verbose: bool, optional
        Print progress or not.
    window_size: int, optional
        Number of consecutive words to perform counting.
        If sentences based analysis used, window size indicate the number of sentences.
    sentences: bool, optional
        If True, documents are list of lists, where each entry in documents
        is a list of sentences from that document. 
        In this case, window size is used as number of sentences, and the matrix is build 

    Returns
    ------
    defaultdict
        word occurance counts.

    """
        
    d = defaultdict(int)
    
    for document in tqdm(documents, total=len(documents), disable=not (verbose)):

        # document sentence based analysis
        if sentences:
            curr_sentences = document
            curr_sentences_combined = []
            tmp = []

            # combine sentences based on window size
            for sent_idx, sent in enumerate(curr_sentences):
                tmp.append(sent)
                if (len(tmp) == window_size) or (sent_idx + 1 == len(curr_sentences)):
                    curr_sentences_combined.append(" ".join(tmp))
                    tmp = []

            # iterate over sentences
            for sent in curr_sentences_combined:
                tokens = sent.split()

                # iterate over words
                for ii in range(len(tokens)):
                    token = tokens[ii]
                    next_token = tokens[ii + 1: len(tokens)]

                    for t in next_token:
                        try:
                            r = V_map[t]
                            c = V_map[token]
                        except Exception:
                            continue
                        
                        key = tuple(sorted([r, c]))
                        d[key] += 1

        # document window size based analysis
        else:
            tokens = document.split()

            # iterate over words
            for ii in range(len(tokens)):
                token = tokens[ii]
                next_token = tokens[ii + 1: min(ii + 1 + window_size, len(tokens))]
                for t in next_token:
                    try:
                        r = V_map[t]
                        c = V_map[token]
                    except Exception:
                        continue
                        
                    key = tuple(sorted([r, c]))
                    d[key] += 1
                    
    return d

def co_occurrence(documents, vocabulary, window_size=20, verbose=True, sentences=False, n_jobs=-1, n_nodes=1, parallel_backend="multiprocessing"):
    """
    Forms co-occurance matrix.

    Parameters
    ----------
    documents: list
        List of documents. Each entry in the list contains text.
        if sentences=True, then documents is a list of lists, where
        each entry in documents is a list of sentences.
    window_size: int, optional
        Number of consecutive words to perform counting.
        If sentences based analysis used, window size indicate the number of sentences.
    vocabulary: list
        List of unqiue words present in the all documents as vocabulary.
    verbose: bool, optional
        Print progress or not.
    sentences: bool, optional
        If True, documents are list of lists, where each entry in documents
        is a list of sentences from that document. 
        In this case, window size is used as number of sentences, and the matrix is build based
        on the sentences. When False, window size is used, and documents is a list of documents.

    Returns
    -------
    M: np.ndarray or sparse CSR matrix
        Co-occurance matrix.

    """
    # turn the vocab into a hashmap for O(1) lookup
    V_map = dict()
    for ii, word in enumerate(vocabulary):
        V_map[word] = ii

    
    if n_nodes > 1 and MPI is None:
        sys.exit("Attempted to use n_nodes>1 but MPI is not available!")
    assert n_nodes > 0, "Number of nodes must be 1 or higher!"
    
    # handle for multiple nodes
    if n_nodes > 1:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        documents_node_chunks = np.array_split(documents, n_nodes)
        documents = documents_node_chunks[rank]
    else:
        comm = None
        rank = -1
    
    # calculate number of chunks
    if n_jobs <= 0:
        n_chunks = multiprocessing.cpu_count()
        n_chunks -= (np.abs(n_jobs) + 1)
    else:
        n_chunks = n_jobs
    
    # chunk the documents to get list of jobs
    document_chunks = list(_chunk_list(documents, n_chunks))
    
    # run counting in parallel
    list_results = Parallel(n_jobs=n_jobs, 
                            verbose=verbose, 
                            backend=parallel_backend)(
            delayed(_co_occurance_parallel_helper)(documents=chunk, 
                                                   verbose=verbose, 
                                                   sentences=sentences,
                                                   V_map=V_map,
                                                   window_size=window_size) for chunk in document_chunks)
    
    # free memory
    del document_chunks
    del documents
    
    cols = list()
    rows = list()
    values = list()

    M = list()
    matrix_dict = defaultdict(int)
    while tqdm(list_results, disable=not (verbose)):
        
        # free memory, get next
        d = list_results.pop(0)
        
        for key, value in tqdm(d.items(), disable=not (verbose)):
            r = key[0]
            c = key[1]
            
            # if diogonal
            if c == r:
                value = value * 2

            # sparse
            matrix_dict[(r, c)] += value

            # non diogonal
            if r != c:
                matrix_dict[(c, r)] += value
                
    # free memory
    del list_results
        
    # if distributed
    if n_nodes > 1:
        
        # prepare COO data for communication
        tensor_data_comm = []
        for key, value in tqdm(matrix_dict.items(), disable=not verbose, total=len(matrix_dict)):
            curr_cords = [float(key[0]), float(key[1])]
            tensor_data_comm.extend(curr_cords)
            tensor_data_comm.append(float(value))
            
        tensor_data_comm = np.array(tensor_data_comm, dtype=float)
        
        # wait for everyone if multiple nodes
        comm.Barrier()
        
        # chunk the list first so that we can communicate the size
        n = 1000000
        tensor_data_chunks = [tensor_data_comm[i:i + n]
                                  for i in range(0, len(tensor_data_comm), n)]
        
        # wait for everyone if multiple nodes
        comm.Barrier()
        
        chunk_sizes = np.array(comm.allgather(len(tensor_data_chunks)))
        maximum_chunk_size = max(chunk_sizes)
        while len(tensor_data_chunks) < maximum_chunk_size:
                tensor_data_chunks.append(np.array([], dtype=float))
                
        if rank == 0:
            all_chunks = []
            
        for ii, chunk in enumerate(tensor_data_chunks):

            # gather the sizes first
            sendcounts = np.array(comm.gather(len(chunk), root=0))

            if rank == 0:
                recvbuf = np.empty(sum(sendcounts), dtype=float)
            else:
                recvbuf = None

            comm.Gatherv(sendbuf=chunk, recvbuf=(recvbuf, sendcounts), root=0)

            if rank == 0:
                all_chunks.extend(recvbuf)
        
        # wait for everyone if multiple nodes
        comm.Barrier()
        
        if rank == 0:
            tensor_data_comm = all_chunks
        else:
            sys.exit(0)
    
        # first combine all elements into single element from all nodes
        matrix_dict = defaultdict(int)
        for x, y, value in tqdm(zip(*[iter(tensor_data_comm)]*3), disable=not verbose, total=len(tensor_data_comm)/3):
            matrix_dict[(int(x), int(y))] += int(value)
         
    # sparse matrix from coordinate and values
    if verbose:
        print("Building sparse matrix from COO matrix...")
    M = sparse.COO(matrix_dict, shape=(len(V_map), len(V_map)))
    M = M.tocsr()
    return M

