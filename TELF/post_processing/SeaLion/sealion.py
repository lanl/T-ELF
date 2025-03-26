# DEPENDENCIES
import numpy as np
from pathlib import Path
import warnings
import os 

# SeaLion Tools
from .tools.get_factorization_results import copy_results
from .tools.get_correlation_matrix import  corr_matrix
from .tools.get_mixing_patterns import mixing_patterns
from .tools.get_H_clustering import H_clustering
from .tools.get_recommendations import recommendations
from .tools.get_recommendations import recommendations_masked
from .tools.get_recommendations import recommendation_graph
from .tools.get_wordclouds import words_probabilities
from .tools.get_wordclouds import word_cloud
from .tools.get_W_plot import W_plot
from .tools.get_W_plot import get_W_sub
from .tools.get_W_plot import W_UMAP
from .tools.get_X import original_X
from .tools.get_X import X_tilda

class SeaLion:

    def __init__(
            self,
            save_path="SeaLion/",
            X=None,
            W=None,
            S=None,
            H=None,
            bu=None,
            bi=None,
            global_mean=0,
            figsize1=None,
            figsize2=None,
            rows=None,
            cols=None,
            rows_name="Features",
            cols_name="Samples",
            num_top_words=50,
            edge_weight_multiplier=1,
            num_top_recommendations=30,
            recommend_probabilities=False,
            factorization_paths=[],
            UNKNOWN_MASK=None,
            KNOWN_MASK=None,
            verbose=True) -> None:

        #
        # SeaLion object variables
        #
        self.save_path = save_path
        
        self.X = X
        self.W = W
        self.S = S
        self.H = H
        
        if bu is None and W is not None:
            self.bu = np.zeros(W.shape[0])
        else:
            self.bu = bu

        if bi is None and H is not None:
            self.bi = np.zeros(H.shape[1])
        else:
            self.bi = bi

        self.global_mean = global_mean
        
        if rows is None:
            self.rows = np.arange(0, X.shape[0], 1).astype("int32")
        else:
            self.rows = np.array(rows)

        if cols is None:
            self.cols = np.arange(0, X.shape[1], 1).astype("int32")
        else:
            self.cols = np.array(cols)

        self.rows_name = rows_name
        self.cols_name = cols_name
        self.figsize1 = figsize1
        self.figsize2 = figsize2

        self.num_top_words = num_top_words
        self.num_top_recommendations = num_top_recommendations
        self.recommend_probabilities = recommend_probabilities

        self.UNKNOWN_MASK = UNKNOWN_MASK
        self.KNOWN_MASK = KNOWN_MASK
        self.factorization_paths = factorization_paths
        self.verbose = verbose
        self.edge_weight_multiplier = edge_weight_multiplier

        #
        # Results path
        #
        if not Path(self.save_path).is_dir():
            Path(self.save_path).mkdir(parents=True)

        #
        # Organize topic paths
        #
        if W is not None:
            K = W.shape[1]
        elif H is not None:
            K = H.shape[0]
        else:
            K = 0
            warnings.warn("W and H were both not passed. Could not create directories for clusters!")
        
        for k in range(K):
            curr_path = os.path.join(self.save_path, f'{k}')
            if not Path(curr_path).is_dir():
                Path(curr_path).mkdir(parents=True)
            
        #
        # Other variables
        #
        self.words = None
        self.probabilities = None
        self.Wsub_name_idx_map = None
        self.W_sub = None
        self.Wsub_mask = None

    def __call__(self):
        if self.verbose:
            print("Starting general post-processing")

        self.get_factorization_results()
        self.get_W_correlation()
        self.get_H_correlation()
        self.get_S_mixing_patterns()
        self.get_H_clustering()
        self.get_words_probabilities()
        self.get_wordclouds()
        self.get_W_plot()
        self.get_W_UMAP()
        self.get_original_data()
        self.get_X_tilda()
        self.get_recommendations()
        self.get_masked_recommendations()
        self.get_recommendations_graph()

        if self.verbose:
            print("Done")

    def get_masked_recommendations(self):
        if self.UNKNOWN_MASK is None or self.KNOWN_MASK is None:
            if self.verbose:
                print("Skipping getting unknown masked recommendations.")
            return
        
        recommendations_masked(
            UNKNOWN_MASK=self.UNKNOWN_MASK,
            KNOWN_MASK=self.KNOWN_MASK,
            W=self.W,
            H=self.H,
            S=self.S,
            bi=self.bi,
            bu=self.bu,
            global_mean=self.global_mean,
            rows=self.rows,
            cols=self.cols,
            save_path=self.save_path,
            num_top_recommendations=self.num_top_recommendations,
            recommend_probabilities=self.recommend_probabilities,
            cols_name=self.cols_name,
            rows_name=self.rows_name
        )
    
    def get_recommendations_graph(self):
        if self.UNKNOWN_MASK is None or self.KNOWN_MASK is None:
            if self.verbose:
                print("Skipping getting recommendations graph.")
            return
        
        recommendation_graph(
            UNKNOWN_MASK=self.UNKNOWN_MASK,
            KNOWN_MASK=self.KNOWN_MASK,
            W=self.W,
            H=self.H,
            S=self.S,
            bi=self.bi,
            bu=self.bu,
            global_mean=self.global_mean,
            rows=self.rows,
            cols=self.cols,
            num_top_recommendations=self.num_top_recommendations,
            recommend_probabilities=self.recommend_probabilities,
            save_path=self.save_path,
            edge_weight_multiplier = self.edge_weight_multiplier
        )

    def get_recommendations(self):
        recommendations(
            W=self.W,
            H=self.H,
            S=self.S,
            bi=self.bi,
            bu=self.bu,
            global_mean=self.global_mean,
            rows=self.rows,
            cols=self.cols,
            save_path=self.save_path,
            num_top_recommendations=self.num_top_recommendations,
            recommend_probabilities=self.recommend_probabilities
        )

    def get_X_tilda(self):
        X_tilda(
            W=self.W,
            H=self.H,
            S=self.S,
            bi=self.bi,
            bu=self.bu,
            global_mean=self.global_mean,
            rows=self.rows,
            cols=self.cols,
            save_path=self.save_path
            )

    def get_original_data(self):
        original_X(X=self.X, rows=self.rows, cols=self.cols, save_path=self.save_path)

    def get_factorization_results(self, factorization_name="factorization"):
        if len(self.factorization_paths) > 0:
            copy_results(
                factorization_paths=self.factorization_paths,
                save_path=self.save_path,
                factorization_name=factorization_name
            )
        else:
            if self.verbose:
                print("Skipping copying the factorization results.")

    def get_W_correlation(self):
        if self.W is None:
            if self.verbose:
                print("Skipping getting W Factor Column Wise Correlation.")
            return
        corr_matrix(A=self.W, name="W", save_path=self.save_path)

    def get_H_correlation(self):
        if self.H is None:
            if self.verbose:
                print("Skipping getting H Factor Row Wise Correlation.")
            return
        corr_matrix(A=self.H.T, name="H", save_path=self.save_path)

    def get_S_mixing_patterns(self):
        if self.S is None:
            if self.verbose:
                print("Skipping getting S mixing matrix patterns")
            return
        mixing_patterns(S=self.S, save_path=self.save_path)

    def get_H_clustering(self):
        if self.H is None:
            if self.verbose:
                print("Skipping getting the H clustering results")
            return
        H_clustering(H=self.H, S=self.S, save_path=self.save_path, cols_name=self.cols_name, cols=self.cols, figsize1=self.figsize1)

    def get_words_probabilities(self):
        if self.W is None:
            if self.verbose:
                print("Skipping getting top words and their probabilities")
            return
        

        self.words, self.probabilities = words_probabilities(W=self.W, 
                                                             save_path=self.save_path, 
                                                             rows=self.rows, 
                                                             rows_name=self.rows_name,
                                                             num_top_words=self.num_top_words)
        
    def get_wordclouds(self):
        if self.W is None:
            if self.verbose:
                    print("Skipping getting top words and their probabilities")
            return
        
        self.words, self.probabilities = words_probabilities(W=self.W, 
                                                             save_path=self.save_path, 
                                                             rows=self.rows, 
                                                             rows_name=self.rows_name,
                                                             num_top_words=self.num_top_words)
        word_cloud(self.words, self.probabilities, self.save_path, max_words=self.num_top_words, background_color='white', format='png')

    
    def get_W_plot(self):
        if self.W is None:
            if self.verbose:
                    print("Skipping getting the W plot")
            return
        self.words, self.probabilities = words_probabilities(W=self.W, 
                                                             save_path=self.save_path, 
                                                             rows=self.rows, 
                                                             rows_name=self.rows_name,
                                                             num_top_words=self.num_top_words)
        
        self.Wsub_name_idx_map, self.W_sub, self.Wsub_mask = get_W_sub(
            words=self.words, 
            probabilities=self.probabilities,
            save_path=self.save_path,
            rows_name=self.rows_name,
            num_top_words=self.num_top_words)
        
        W_plot(
            W_sub=self.W_sub,
            Wsub_mask=self.Wsub_mask,
            Wsub_name_idx_map=self.Wsub_name_idx_map,
            num_top_words=self.num_top_words, 
            rows_name=self.rows_name, 
            save_path=self.save_path,
            figsize2=self.figsize2
            )
        
    def get_W_UMAP(self, args={}):
        if self.W is None:
            if self.verbose:
                    print("Skipping getting the W UMAP plot")
            return
        
        self.words, self.probabilities = words_probabilities(W=self.W, 
                                                             save_path=self.save_path, 
                                                             rows=self.rows, 
                                                             rows_name=self.rows_name,
                                                             num_top_words=self.num_top_words)
        
        self.Wsub_name_idx_map, self.W_sub, self.Wsub_mask = get_W_sub(
            words=self.words, 
            probabilities=self.probabilities,
            save_path=self.save_path,
            rows_name=self.rows_name,
            num_top_words=self.num_top_words)
        
        W_UMAP(W_sub=self.W_sub,
               Wsub_name_idx_map=self.Wsub_name_idx_map,
               save_path=self.save_path,
               num_top_words=self.num_top_words,
               args=args)