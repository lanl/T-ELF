from itertools import combinations
import networkx as nx
import pandas as pd

def create_authors_graph(df, col):
    G = nx.Graph()
    for index, row in df.iterrows():
        if pd.isna(row[col]):
            continue

        # add edges between every pair of authors for given paper
        authors = str(row[col]).split(';')
        for a, b in combinations(authors, 2):
            if G.has_edge(a, b): # increase weight by 1 if edge already exists
                G[a][b]['weight'] += 1
            else: # new edge, set weight = 1
                G.add_edge(a, b, weight=1)
    return G
