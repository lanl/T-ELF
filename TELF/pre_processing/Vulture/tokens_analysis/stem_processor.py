from nltk.stem import PorterStemmer
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from TELF.pre_processing.Vulture.tokens_analysis.levenstein import compare_keys

"""
SAMPLE USAGE
----------
stem_processor = StemProcessor(vocabulary)
subs_stemed, new_vocabulary = stem_processor.build_vocab_stem_subs()
"""
class StemProcessor:
    def __init__(self, vocabulary, suffixes=None):
        """
        Store values for processing in functions

        Parameters
        ----------
        vocabulary : list
            words from the corpus
        suffixes : list
            common suffixes in english
        """
        SUFFIXES = ['acity', 'ation', 'ative', 'cracy', 'craft', 'esque', 'able', 
                    'ance', 'ancy', 'cide', 'ence', 'ency', 'hood', 'ible', 'less', 
                    'like', 'ment', 'ness', 'ship', 'sion', 'ster', 'tion', 'ward', 
                    'ware', 'wise', 'acy', 'ant', 'ary', 'ate', 'dom', 'ent', 'ern', 
                    'ese', 'ess', 'est', 'ful', 'ian', 'ice', 'ify', 'ing', 'ion', 
                    'ish', 'ism', 'ist', 'ity', 'ive', 'ize', 'ory', 'ous', 'ac', 
                    'al', 'ar', 'ed', 'ee', 'en', 'er', 'fy', 'ic', 'ly', 'or', 'ty', 
                    'y']
        if suffixes:
            self.suffixes = sorted(suffixes, key=len, reverse=True)
        else:
            self.suffixes = SUFFIXES
        self.vocabulary = vocabulary

    def strip_suffixes(self, word):
        """
        Removes all suffixes, longest to shorest

        Parameters
        ----------
        word : str
            unified variants map to shortest variant
    
        Returns
        -------
        word :  str
            word without suffixes
        """
        for suffix in self.suffixes:
            if word.endswith(suffix):
                return word[:-len(suffix)]
        return word

    def unify_common_stems(self, vocab_stems, similarity_threshold=0.9, min_word_length=5, n_jobs=None):
        """
        finds stems that are the same without endings

        Parameters
        ----------
        vocab_stems : dict (str:str)
            unified variants map to shortest variant
        similarity_threshold : float
            similarity cutoff
        min_word_length : int
            only consider words meeting this length
        n_jobs : int
            number of concurrent jobs
    
        Returns
        -------
        vocab_stems : dict (str:str)
            unified variants map to shortest variant
        """
        def compare_stems(stem_pair):
            stem_i, stem_j = stem_pair
            if len(stem_i) > min_word_length and len(stem_j) > min_word_length:
                compare_i = self.strip_suffixes(stem_i)
                compare_j = self.strip_suffixes(stem_j)
                similar, _ = compare_keys(compare_i, compare_j, threshold=similarity_threshold)
                if similar:
                    return (stem_i, stem_j)
            return None

        stems = list(vocab_stems.keys())
        stem_pairs = [(stems[i], stems[j]) for i in range(len(stems)) for j in range(i + 1, len(stems)) if stems[i][0] == stems[j][0]]
        similar = []

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(tqdm(executor.map(compare_stems, stem_pairs), total=len(stem_pairs)))

        similar = [result for result in results if result is not None]

        seen = {}
        for stem_i, stem_j in similar:
            shortest_stem = min(stem_i, stem_j, key=len)
            longest_stem = stem_j if shortest_stem == stem_i else stem_i

            destination_map = seen.get(longest_stem, shortest_stem)
            if longest_stem in vocab_stems:
                if destination_map in vocab_stems:
                    vocab_stems[destination_map]['src'].extend(vocab_stems.pop(longest_stem)['src'])
                else:
                    vocab_stems[destination_map] = {'src': vocab_stems.pop(longest_stem)['src'], 'dest': vocab_stems[destination_map]}
                seen[longest_stem] = shortest_stem

        return vocab_stems

    def build_stem_map(self):
        """
        Stems vocabulary map, ununified

        Returns
        -------
        vocab_stems : dict (str:str)
            variants map to shortest variant, ununified
        """
        ps = PorterStemmer()
        vocab_stems = {}
        for word in self.vocabulary:
            stem = ps.stem(word)
            if stem in vocab_stems:
                vocab_stems[stem]['src'].append(word)
            else:
                vocab_stems[stem] = {'src': [word], 'dest': word}

            shortest_word = min(vocab_stems[stem]['src'], key=len)
            vocab_stems[stem]['dest'] = shortest_word

        return vocab_stems

    def build_vocab_stem_subs(self):

        """
        Stems vocabulary, constructs map of all variants to the shorstest variant.

        Returns
        -------
        subs_stemed : dict (str:str)
            variants map to shortest variant
        shortened_vocabulary : list
            new vocabulary post-consolidation
        """
        subs_stemed = {}
        vocab_stems = self.build_stem_map()
        vocab_stems = self.unify_common_stems(vocab_stems)
        shortened_vocabulary = set()

        for stem, info in vocab_stems.items():
            destination_word = info['dest']
            shortened_vocabulary.add(destination_word)
            for src in info['src']:
                if src != destination_word:
                    subs_stemed[src] = destination_word

        return subs_stemed, list(shortened_vocabulary)
