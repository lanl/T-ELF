import re
import warnings
import itertools
import networkx as nx
 

import pandas as pd, os
from TELF.pre_processing.Vulture.tokens_analysis.top_words import get_top_words
from TELF.pre_processing.Vulture.modules import VultureModuleBase



class AcronymCleaner(VultureModuleBase):
    """
    Generates acronyms from the docuemnts from n-grams. Terms can be excluded from the acronyms and custom acronyms can be added.

    Parameters
    ----------
    df : pd.DataFrame, list, dict
        list or dictionary of documents. 
        If dictionary, keys are the document IDs, values are the text.
        if df, need to pass a column.
    grams_savepath : str, optional
        if present, saves the acronyms and grams at the location
    exclude_terms : int, optional
        keywords to search acronyms and grams for. If found, the gram is excluded.
    grams_to_n : bool, optional
        number of grams to search for acronyms. Always starts at 2. This param is the stop
    column : str, optional
        required if passed dataype is a pd.DataFrame
    custom_acronyms : pd.DataFrame, optional
        If present, these (term, acronym, weight) row triplets are added to the acronym file.
    weight: int, optional
        specifies the output weight for the acronyms
    sort_wordcount: bool, optional
        should the words be sorted by the number of words that compsoe the acronym (the gram count)
        Useful if acronyms are going to be substituted into Vulture, largest terms are subbed first.
    save_grams: bool, optional
        should the grams that are generated to serch for the acronyms be saved

    Returns
    -------
    pd.DataFrame
        Acronyms aggregated

    """
    def __init__(self, merge_sim_threshold=0.9, verbose=False, frozen=None, **kwargs):
        super().__init__(frozen, **kwargs)  # initialize the base class with the preserve
        # self.pid = os.getpid()
        self.merge_sim_threshold = merge_sim_threshold

    def __call__(self, document ):
        return self.run(document, verbose=False)
        
    def run(self, 
            document, 
            verbose=None,

            acronyms_savepath: str="./acronyms", 
            exclude_terms: list=None, 
            grams_to_n: int=7, 
            column: str="clean_title_abstract", 
            custom_acronyms: pd.DataFrame=None, 
            weight: int=2,
            sort_wordcount: bool=True,
         
            ):
        print("run sub starts: ", document)

        
        return_acronyms = False
        # waterfall order of saving
        if verbose == True and self.verbose == True:
            returning_acronyms = True
        elif verbose == False: # forced false from __call__ as in vulture steps
            returning_acronyms = False
        elif verbose == None and self.verbose == True:
            returning_acronyms = True
        acronym_subs = pd.DataFrame()
       
        if isinstance(document, tuple):
            documents = {document[0]: document[1]} 
            doc_id = document[0] 
        elif isinstance(document, str):
            documents = {0: document} 
            doc_id =  0
        elif isinstance(document, pd.DataFrame):
            if column not in document.columns:
                raise ValueError(f"Column not in dataframe:{column}")
            documents = document[column].to_dict()
            
        elif isinstance(document, dict):
            documents = document
        elif isinstance(document, list):
            documents = {i: document[i] for i in range(len(document))}
        else:
            raise ValueError(f"Incompatible datatype passed to in find_acronyms: {type(document)}")
        print("run sub starts: ", documents)
        
        grams_container = []
        only_acronyms = []
        next_acronyms = []
        for n in range(grams_to_n):
            grams_container.append( get_top_words(  documents, 
                                                    top_n=99999, n_gram=n+1, 
                                                    verbose=True, filename=None) 
                                )
            if exclude_terms:
                grams_container[n] = grams_container[n][~grams_container[n]['word'].apply(lambda row: any(item in row for item in exclude_terms))]
            # only_acronyms.append(self.find_acronyms_helper(grams_container[n], keep_only_acronyms=1) )

            if len(grams_container[n]) > 0 :
                next_acronyms = self.find_acronyms_helper(grams_container[n], keep_only_acronyms=1)
            if len(next_acronyms) > 0:
                    only_acronyms.append( next_acronyms)
        
        if isinstance(custom_acronyms, pd.DataFrame):
            only_acronyms.append(custom_acronyms)
        
        if only_acronyms:
            acronym_subs = pd.concat(only_acronyms)
        

        # NO SUBS, acronym return
        if acronyms_savepath and returning_acronyms:
            if acronym_subs.empty: 
                print("empty 1")
                return 
            acronym_subs['weight'] = weight

            # Headers are added when the terms are not sorted by word frequency
            use_header=True
            if sort_wordcount:
                acronym_subs['word_count'] = acronym_subs['word'].apply(lambda x: len(x.split()))
                acronym_subs = acronym_subs.sort_values(by='word_count',  ascending=False).drop('word_count', axis=1)
                use_header=False

            if not os.path.exists(acronyms_savepath):
                os.makedirs(acronyms_savepath)

            acronym_path = os.path.join(acronyms_savepath, f'acronyms.csv')
            acronym_subs.to_csv(acronym_path, index=False, header=use_header)

            all_grams_all_n = []
            for n in range(grams_to_n):
                all_grams_all_n.append(self.find_acronyms_helper(grams_container[n]))
                all_grams_all_n[n].to_csv(os.path.join(acronyms_savepath,f'bow_{str(n+1)}_grams.csv'), index=False)
                only_acronyms[n].to_csv(os.path.join(acronyms_savepath,f'ACRONYM_bow_{str(n+1)}_grams.csv'), index=False) 
        
            return acronym_subs
        
        # SUBS, no acronym returns, single document expected
        else:

            print("single sub else: ", document[0])
            if acronym_subs.empty: 
                print("empty 2")

                return document
            
            if len(documents) > 1:
                raise ValueError("Docuemnts need to be single text for substitution. If multiple substitution is needed,"+\
                                " run AcronymCleaner.run(verbose = True) outside of vulture clean. Then use vulture substitutions "+\
                                f"on the returned acronyms. Current (type, size): {(type(documents), len(documents))}")
            # No SUBS
            if len(only_acronyms) < 1:
                return document
            
            search_terms = acronym_subs['word']
            sub_terms = acronym_subs['substitution']
            print("before words documents", documents)
            print("before words doc_id", doc_id)

            words = documents[doc_id].split()
            replaced_words = [sub_terms[search_terms.index(word)] if word in search_terms else word for word in words]
            replaced_text = ' '.join(replaced_words)
            return (doc_id, replaced_text)



    def find_acronyms_helper(self, df, ignore_acs=[], ignore_gram_part=[], keep_only_acronyms=0):
        """
        KEY --> keep_only_acronyms: -1 = only non-acronyms
                                    0 = acronyms + non-acronyms
                                    1 = only acronyms
        """                             
        df_new = pd.DataFrame()
        all_subs = [];         all_words = [];         all_tf = [];         all_df = []
        
        for word,tf, df in zip(df['word'],df['tf'], df['df']):
            parts = word.split()
            grams_without_end = parts[:-1]        
            acronym = "".join([piece[0] for piece in grams_without_end])
            last_gram = parts[-1]
            
            exact_acronym = True # Check to see if all preceeding parts' first letters match the last gram entirely
            if len(acronym) == len(last_gram):
                for i, letter in enumerate(acronym):
                    if letter != last_gram[i]:
                        exact_acronym = False
                        break
            else:
                exact_acronym = False
                
            if exact_acronym and keep_only_acronyms > -1:
                sub = "_".join(grams_without_end)
                need_to_ignore = False
                for ignore in ignore_acs:
                    if sub in ignore:
                        need_to_ignore = True
                        break
                        
                if not need_to_ignore:
                    for ignore in ignore_gram_part:
                        if ignore in sub:
                            need_to_ignore = True
                            break 
                        
                if not need_to_ignore:       
                    word_no_acronym = " ".join(grams_without_end)
                    all_subs.append(sub);                 all_words.append(word);     all_tf.append(tf);    all_df.append(df)
                    all_words.append(last_gram);          all_subs.append(sub);       all_tf.append(tf);    all_df.append(df)
                    all_words.append(word_no_acronym);    all_subs.append(sub);       all_tf.append(tf);    all_df.append(df)
                    
            else:
                if keep_only_acronyms < 1:  
                    need_to_ignore = False
                    sub = "_".join(parts)
                    for ignore in ignore_acs:
                        if sub in ignore:
                            need_to_ignore = True
                            break
                            
                    if not need_to_ignore:
                        for ignore in ignore_gram_part:
                            if ignore in sub:
                                need_to_ignore = True
                                break 
                    
                    if not need_to_ignore:
                        all_subs.append(sub);             all_words.append(word);     all_tf.append(tf);    all_df.append(df)
        
        df_new['word'] = all_words
        df_new['substitution'] = all_subs
        df_new['tf'] = all_tf
        df_new['df'] = all_df
        return df_new

    # def make_acronym_subs(self, text:str, permutations_path=None, grams_1_to_n=7):
    #     permutations_path = self.path_passed_or_env(permutations_path, 'permutations_path')
         
    #     remove_from_vocab = []
    #     with open(permutations_path) as f:
    #         for line in f:
    #             remove_from_vocab.append(line.strip().lower())
        
    #     grams_container = []
    #     only_acronyms = []
    #     for n in range(grams_1_to_n):
    #         grams_container.append( get_top_words(  [text], 
    #                                                 top_n=99999, n_gram=n+1, 
    #                                                 verbose=False, filename=None) 
    #                               )
    #         grams_container[n] = grams_container[n][~grams_container[n]['word'].apply(lambda row: any(item in row for item in remove_from_vocab))]
         
    #         if len(grams_container[n]) > 0 :
    #             next_acronyms = self.make_sub_col_from_gram(grams_container[n], keep_only_acronyms=1)
    #             if len(next_acronyms) > 0:
    #                 only_acronyms.append( next_acronyms)
                
    #     if len(only_acronyms) < 1:
    #         return None, text
    #     acronym_subs = pd.concat(only_acronyms)
    #     acronym_subs['weight'] = 2

    #     search_terms = acronym_subs['word']
    #     sub_terms = acronym_subs['subs']
    #     words = text.split()
    #     replaced_words = [sub_terms[search_terms.index(word)] if word in search_terms else word for word in words]
    #     replaced_text = ' '.join(replaced_words)

    #     return acronym_subs, replaced_text


    # def check_unique_or_concatenate(self, series):
    #     if series.nunique() == 1:  
    #         return series.iloc[0]
    #     else:
    #         all_unique_components = set()
            
    #         for value in series.unique():
    #             all_unique_components.update(value.split(' | '))
    #         return ' | '.join(sorted(all_unique_components)) 

    # def consolidate_acronyms(self,all_acronyms):
    #     combined_df = pd.concat(all_acronyms, ignore_index=True)
    #     aggregated_acronyms = combined_df.groupby(['word','subs'], as_index=False).agg({
    #         'subs': self.check_unique_or_concatenate,
    #         'tf': 'sum',
    #         'df': 'sum',
    #     })
    #     final_acronyms_consolidated = aggregated_acronyms.sort_values(by='df', ascending=False).reset_index(drop=True)
    #     return final_acronyms_consolidated

    # def clean_acronyms_sub_df(self, material_path, column="clean_title_abstract"):
    #     df = pd.read_csv(material_path, lineterminator='\n')
    #     text_data = df[column].tolist()
    #     all_acronyms = []
    #     new_texts = []
    #     for text in tqdm(text_data[:]):
    #         current_acs, subbed_text = self.make_acronym_subs(text)
    #         if type(current_acs) != type(None):
    #             all_acronyms.append(current_acs)
            
    #         new_texts.append(subbed_text)
       

    #     df['clean_subs'] = new_texts
    #     # SAVE TO ORIGINAL DATA
    #     df.to_csv(material_path, index=False) 
    #     consolidated_acronyms = self.consolidate_acronyms(all_acronyms)
    #     return consolidated_acronyms