#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 08:28:49 2022

@author: maksim
"""
from scipy.spatial.distance import cosine
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy


def H_clustering(H, verbose=False):
    """
    Performs H-clustering, and gathers cluster information.

    Parameters
    ----------
    H: np.ndarray or scipy.sparse.csr_matrix
        H matrix from NMF
    verbose: bool, default is False
        If True, shows the progress.

    Returns
    -------
    (clusters_information, centroid_similarities): tuple of dict and list
    Dictionary carrying information for each cluster, 
    and dictionary carrying information for each document.
    """

    # hyper-parameter check
    assert scipy.sparse.issparse(H) or \
        type(H) == np.ndarray, \
        "H type is not supported. H type: " + str(type(H)) + "\n" \
        "H should be type scipy.sparse.csr_matrix or np.ndarray."

    assert type(verbose) is bool, "verbose should be type bool!"

    # begin
    cluster_assignments = np.argmax(H, axis=0)
    total_documents = H.shape[1]
    clusters_information = {}
    documents_information = {}

    # clusters information
    for cluster in tqdm(set(cluster_assignments), disable=not verbose):

        # get the documents in the current cluster
        docs_in_cluster = np.argwhere(cluster_assignments == cluster).flatten()

        # calculate the cluster information
        num_docs_in_cluster = len(docs_in_cluster)
        percent_docs_in_cluster = num_docs_in_cluster / total_documents

        # calculate the centroid information
        coordinates_in_cluster = H[:, docs_in_cluster]
        cluster_centroid = np.sum(coordinates_in_cluster, axis=1) / num_docs_in_cluster

        # save the information
        clusters_information[cluster] = {
            "count": num_docs_in_cluster,
            "percent": percent_docs_in_cluster,
            "centroid": cluster_centroid
        }

    # documents information, go through document's coordinates and its index
    for doc_idx, doc_coord in tqdm(enumerate(H.transpose()),
                                   disable=not verbose, total=H.shape[1]):

        # cluster that the document belongs to
        doc_cluster = cluster_assignments[doc_idx]

        # centroid of the cluster that the current document belongs to
        cluster_centroid = clusters_information[doc_cluster]["centroid"]

        # similarity of the coordinate of the document to the cluster centroid
        doc_sim_centroid = 1 - cosine(doc_coord, cluster_centroid)
        documents_information[doc_idx] = {
            "similarity_to_cluster_centroid": doc_sim_centroid,
            "cluster": doc_cluster
        }

    return (clusters_information, documents_information)


def plot_H_clustering(H, name="filename"):
    """
    Plots the centroids of the H-clusters

    Parameters
    ----------
    H: np.ndarray or scipy.sparse.csr_matrix
        H matrix from NMF
    name: File name to save

    Returns
    -------
    Matplotlib plots
    """
    labels = np.argmax(H, axis=0)
    uniques = np.unique(labels)
    for i, l in enumerate(uniques):
        fig = plt.figure(figsize=(6, 3*uniques.shape[0]))
        cluster = H[:, labels == l]
        cluster_means = np.mean(cluster, axis=1)
        cluster_stds = np.std(cluster, axis=1)
        ax = plt.subplot(uniques.shape[0], 1, i+1)
        plt.bar(np.arange(H.shape[0]), cluster_means, color=(0.2, 0.4, 0.6, 0.6))
        plt.title(f'{name} cluster {i}')
        plt.tight_layout()
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{name}_{i}.png', bbox_inches=extent.expanded(1.2, 1.3), dpi=200)
        plt.close()

    return fig
